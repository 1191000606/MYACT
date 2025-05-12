#!/home/lin/software/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""

import torch
import numpy as np
import os
import pickle
import argparse
from einops import rearrange

from utils import compute_dict_mean, set_seed, detach_dict  # helper functions
from policy import ACTPolicy, CNNMLPPolicy, DiffusionPolicy
import collections
from collections import deque

import rospy
from std_msgs.msg import Header
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState, Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import time
import threading
import math


import sys

sys.path.append("./")

task_config = {"camera_names": ["cam_high", "cam_left_wrist", "cam_right_wrist"]}

inference_thread = None
inference_lock = threading.Lock()
inference_actions = None
inference_timestep = None


def actions_interpolation(args, pre_action, actions, stats):
    steps = np.concatenate((np.array(args.arm_steps_length), np.array(args.arm_steps_length)), axis=0)
    pre_process = lambda s_qpos: (s_qpos - stats["qpos_mean"]) / stats["qpos_std"]
    post_process = lambda a: a * stats["qpos_std"] + stats["qpos_mean"]
    result = [pre_action]
    post_action = post_process(actions[0])
    # print("pre_action:", pre_action[7:])
    # print("actions_interpolation1:", post_action[:, 7:])
    max_diff_index = 0
    max_diff = -1
    for i in range(post_action.shape[0]):
        diff = 0
        for j in range(pre_action.shape[0]):
            if j == 6 or j == 13:
                continue
            diff += math.fabs(pre_action[j] - post_action[i][j])
        if diff > max_diff:
            max_diff = diff
            max_diff_index = i

    for i in range(max_diff_index, post_action.shape[0]):
        step = max([math.floor(math.fabs(result[-1][j] - post_action[i][j]) / steps[j]) for j in range(pre_action.shape[0])])
        inter = np.linspace(result[-1], post_action[i], step + 2)
        result.extend(inter[1:])
    while len(result) < args.chunk_size + 1:
        result.append(result[-1])
    result = np.array(result)[1 : args.chunk_size + 1]
    # print("actions_interpolation2:", result.shape, result[:, 7:])
    result = pre_process(result)
    result = result[np.newaxis, :]
    return result


def get_model_config(args):
    # 设置随机种子，你可以确保在相同的初始条件下，每次运行代码时生成的随机数序列是相同的。
    set_seed(1)

    # 如果是ACT策略
    # fixed parameters
    if args.policy_class == "ACT":
        policy_config = {
            "lr": args.lr,
            "lr_backbone": args.lr_backbone,
            "backbone": args.backbone,
            "masks": args.masks,
            "weight_decay": args.weight_decay,
            "dilation": args.dilation,
            "position_embedding": args.position_embedding,
            "loss_function": args.loss_function,
            "chunk_size": args.chunk_size,  # 查询
            "camera_names": task_config["camera_names"],
            "use_depth_image": args.use_depth_image,
            "use_robot_base": args.use_robot_base,
            "kl_weight": args.kl_weight,  # kl散度权重
            "hidden_dim": args.hidden_dim,  # 隐藏层维度
            "dim_feedforward": args.dim_feedforward,
            "enc_layers": args.enc_layers,
            "dec_layers": args.dec_layers,
            "nheads": args.nheads,
            "dropout": args.dropout,
            "pre_norm": args.pre_norm,
        }
    elif args.policy_class == "CNNMLP":
        policy_config = {
            "lr": args.lr,
            "lr_backbone": args.lr_backbone,
            "backbone": args.backbone,
            "masks": args.masks,
            "weight_decay": args.weight_decay,
            "dilation": args.dilation,
            "position_embedding": args.position_embedding,
            "loss_function": args.loss_function,
            "chunk_size": 1,  # 查询
            "camera_names": task_config["camera_names"],
            "use_depth_image": args.use_depth_image,
            "use_robot_base": args.use_robot_base,
        }

    elif args.policy_class == "Diffusion":
        policy_config = {
            "lr": args.lr,
            "lr_backbone": args.lr_backbone,
            "backbone": args.backbone,
            "masks": args.masks,
            "weight_decay": args.weight_decay,
            "dilation": args.dilation,
            "position_embedding": args.position_embedding,
            "loss_function": args.loss_function,
            "chunk_size": args.chunk_size,  # 查询
            "camera_names": task_config["camera_names"],
            "use_depth_image": args.use_depth_image,
            "use_robot_base": args.use_robot_base,
            "observation_horizon": args.observation_horizon,
            "action_horizon": args.action_horizon,
            "num_inference_timesteps": args.num_inference_timesteps,
            "ema_power": args.ema_power,
        }
    else:
        raise NotImplementedError

    config = {
        "ckpt_dir": args.ckpt_dir,
        "ckpt_name": args.ckpt_name,
        "ckpt_stats_name": args.ckpt_stats_name,
        "episode_len": args.max_publish_step,
        "state_dim": args.state_dim,
        "policy_class": args.policy_class,
        "policy_config": policy_config,
        "temporal_agg": args.temporal_agg,
        "camera_names": task_config["camera_names"],
    }
    return config


def make_policy(policy_class, policy_config):
    if policy_class == "ACT":
        policy = ACTPolicy(policy_config)
    elif policy_class == "CNNMLP":
        policy = CNNMLPPolicy(policy_config)
    elif policy_class == "Diffusion":
        policy = DiffusionPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def get_image(observation, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(observation["images"][cam_name], "h w c -> c h w")

        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image


def get_depth_image(observation, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_images.append(observation["images_depth"][cam_name])
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image


def inference_process(args, config, ros_operator, policy, stats, t, pre_action):
    global inference_lock
    global inference_actions
    global inference_timestep
    print_flag = True
    pre_pos_process = lambda s_qpos: (s_qpos - stats["qpos_mean"]) / stats["qpos_std"]
    pre_action_process = lambda next_action: (next_action - stats["action_mean"]) / stats["action_std"]
    rate = rospy.Rate(args.publish_rate)
    while True and not rospy.is_shutdown():
        result = ros_operator.get_frame()
        if not result:
            if print_flag:
                print("syn fail")
                print_flag = False
            rate.sleep()
            continue
        print_flag = True
        (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth, puppet_arm_left, puppet_arm_right, robot_base) = result
        obs = collections.OrderedDict()
        image_dict = dict()

        image_dict[config["camera_names"][0]] = img_front
        image_dict[config["camera_names"][1]] = img_left
        image_dict[config["camera_names"][2]] = img_right

        obs["images"] = image_dict

        if args.use_depth_image:
            image_depth_dict = dict()
            image_depth_dict[config["camera_names"][0]] = img_front_depth
            image_depth_dict[config["camera_names"][1]] = img_left_depth
            image_depth_dict[config["camera_names"][2]] = img_right_depth
            obs["images_depth"] = image_depth_dict

        obs["qpos"] = np.concatenate((np.array(puppet_arm_left.position), np.array(puppet_arm_right.position)), axis=0)
        obs["qvel"] = np.concatenate((np.array(puppet_arm_left.velocity), np.array(puppet_arm_right.velocity)), axis=0)
        obs["effort"] = np.concatenate((np.array(puppet_arm_left.effort), np.array(puppet_arm_right.effort)), axis=0)
        if args.use_robot_base:
            obs["base_vel"] = [robot_base.twist.twist.linear.x, robot_base.twist.twist.angular.z]
            obs["qpos"] = np.concatenate((obs["qpos"], obs["base_vel"]), axis=0)
        else:
            obs["base_vel"] = [0.0, 0.0]
        # qpos_numpy = np.array(obs['qpos'])

        # 归一化处理qpos 并转到cuda
        qpos = pre_pos_process(obs["qpos"])
        qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
        # 当前图像curr_image获取图像
        curr_image = get_image(obs, config["camera_names"])
        curr_depth_image = None
        if args.use_depth_image:
            curr_depth_image = get_depth_image(obs, config["camera_names"])
        start_time = time.time()
        all_actions = policy(curr_image, curr_depth_image, qpos)
        end_time = time.time()
        print("model cost time: ", end_time - start_time)
        inference_lock.acquire()
        inference_actions = all_actions.cpu().detach().numpy()
        if pre_action is None:
            pre_action = obs["qpos"]
        # print("obs['qpos']:", obs['qpos'][7:])
        print("obs['qpos']:", obs["qpos"])

        if args.use_actions_interpolation:
            inference_actions = actions_interpolation(args, pre_action, inference_actions, stats)
        inference_timestep = t
        inference_lock.release()
        break


def model_inference(args, config, ros_operator, save_episode=True):
    global inference_lock
    global inference_actions
    global inference_timestep
    global inference_thread
    set_seed(1000)

    # 1 创建模型数据  继承nn.Module
    policy = make_policy(config["policy_class"], config["policy_config"])
    # print("model structure\n", policy.model)

    # 2 加载模型权重
    ckpt_path = os.path.join(config["ckpt_dir"], config["ckpt_name"])
    state_dict = torch.load(ckpt_path)
    new_state_dict = {}
    for key, value in state_dict.items():
        if key in ["model.is_pad_head.weight", "model.is_pad_head.bias"]:
            continue
        if key in ["model.input_proj_next_action.weight", "model.input_proj_next_action.bias"]:
            continue
        new_state_dict[key] = value
    loading_status = policy.deserialize(new_state_dict)
    if not loading_status:
        print("ckpt path not exist")
        return False

    # 3 模型设置为cuda模式和验证模式
    policy.cuda()
    policy.eval()

    # 4 加载统计值
    stats_path = os.path.join(config["ckpt_dir"], config["ckpt_stats_name"])
    # 统计的数据  # 加载action_mean, action_std, qpos_mean, qpos_std 14维
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    # 数据预处理和后处理函数定义
    pre_process = lambda s_qpos: (s_qpos - stats["qpos_mean"]) / stats["qpos_std"]
    post_process = lambda a: a * stats["qpos_std"] + stats["qpos_mean"]

    max_publish_step = config["episode_len"]
    chunk_size = config["policy_config"]["chunk_size"]

    # 发布基础的姿态
    left0 = [-0.00133514404296875, 0.00209808349609375, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, 3.557830810546875]
    right0 = [-0.00133514404296875, 0.00438690185546875, 0.034523963928222656, -0.053597450256347656, -0.00476837158203125, -0.00209808349609375, 3.557830810546875]
    left1 = [-0.00133514404296875, 0.00209808349609375, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, -0.3393220901489258]
    right1 = [-0.00133514404296875, 0.00247955322265625, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, -0.3397035598754883]

    ros_operator.puppet_arm_publish_continuous(left0, right0)
    input("Enter any key to continue :")
    ros_operator.puppet_arm_publish_continuous(left1, right1)
    action = None
    # 推理
    with torch.inference_mode():
        while True and not rospy.is_shutdown():
            # 每个回合的步数
            t = 0
            max_t = 0
            rate = rospy.Rate(args.publish_rate)
            if config["temporal_agg"]:
                all_time_actions = np.zeros([max_publish_step, max_publish_step + chunk_size, config["state_dim"]])
            while t < max_publish_step and not rospy.is_shutdown():
                # start_time = time.time()
                # query policy
                if config["policy_class"] == "ACT":
                    if t >= max_t:
                        pre_action = action
                        inference_thread = threading.Thread(target=inference_process, args=(args, config, ros_operator, policy, stats, t, pre_action))
                        inference_thread.start()
                        inference_thread.join()
                        # inference_process(args, config, ros_operator,policy, stats, t, pre_action)
                        inference_lock.acquire()
                        if inference_actions is not None:
                            inference_thread = None
                            all_actions = inference_actions
                            inference_actions = None
                            max_t = t + args.pos_lookahead_step
                            if config["temporal_agg"]:
                                all_time_actions[[t], t : t + chunk_size] = all_actions
                        inference_lock.release()
                    if config["temporal_agg"]:
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = np.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = exp_weights[:, np.newaxis]
                        raw_action = (actions_for_curr_step * exp_weights).sum(axis=0, keepdims=True)
                    else:
                        if args.pos_lookahead_step != 0:
                            raw_action = all_actions[:, t % args.pos_lookahead_step]
                        else:
                            raw_action = all_actions[:, t % chunk_size]
                else:
                    raise NotImplementedError
                action = post_process(raw_action[0])
                left_action = action[:7]  # 取7维度
                right_action = action[7:14]
                left_action[-1:] *= 13
                right_action[-1:] *= 13
                # left_action[-1:] = 0.5
                ros_operator.puppet_arm_publish(left_action, right_action)  # puppet_arm_publish_continuous_thread
                if args.use_robot_base:
                    vel_action = action[14:16]
                    ros_operator.robot_base_publish(vel_action)
                t += 1
                # end_time = time.time()
                # print("publish: ", t)
                # print("time:", end_time - start_time)
                # print("left_action:", left_action)
                # print("right_action:", right_action)
                rate.sleep()


def main():
    args = get_arguments()
    ros_operator = RosOperator(args)
    config = get_model_config(args)
    import json

    json.dump(config, open("args.json", "w"), indent=4)
    model_inference(args, config, ros_operator, save_episode=True)


if __name__ == "__main__":
    main()
# python act/inference.py --ckpt_dir ~/train0314/
