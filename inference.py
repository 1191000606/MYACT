import collections
import math
import threading
import time
import numpy as np
import torch
import yaml
from aloha import RosOperator
from model import ACT
import rospy
from einops import rearrange

qpos_mean = [-1.31453339e-06, 1.02282807e-01, 5.46503216e-02, -1.45520553e-01, -1.09127037e-01, -5.24381287e-02, -1.43856155e-02, 1.59340277e-01, 7.98785985e-01, 5.86473405e-01, -8.01886022e-02, -3.14663380e-01, 8.85170102e-02, 1.37638882e-01]

qpos_std = [0.01594464, 0.04671929, 0.01, 0.01, 0.01, 0.01, 0.01, 0.28153193, 0.62630606, 0.46702683, 0.25290582, 0.20176646, 0.13189976, 0.16071655]

normalize = lambda x: (x - qpos_mean) / qpos_std
denormalize = lambda x: x * qpos_std + qpos_mean

inference_lock = threading.Lock()
inference_actions = None
inference_timestep = None
inference_thread = None

def inference_process(config, ros_operator, model, t, pre_action):
    global inference_lock
    global inference_actions
    global inference_timestep
    print_flag = True

    rate = rospy.Rate(config["publish_rate"])

    while True and not rospy.is_shutdown():
        result = ros_operator.get_frame()
        if not result:
            if print_flag:
                print("syn fail")
                print_flag = False
            rate.sleep()
            continue
        print_flag = True

        img_front, img_left, img_right, puppet_arm_left, puppet_arm_right = result

        qpos = np.concatenate((np.array(puppet_arm_left.position), np.array(puppet_arm_right.position)), axis=0)
        qpos = denormalize(qpos)
        qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)

        curr_images = []
        for image in [img_front, img_left, img_right]:
            curr_image = rearrange(image, "h w c -> c h w")
            curr_images.append(curr_image)
        
        curr_image = np.stack(curr_images, axis=0)
        curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)

        start_time = time.time()
        all_actions = model(curr_image, qpos)
        end_time = time.time()
        print("model cost time: ", end_time - start_time)
        inference_lock.acquire()
        inference_actions = all_actions.cpu().detach().numpy()
        if pre_action is None:
            pre_action = qpos.cpu().detach().numpy()

        if config["use_actions_interpolation"]:
            inference_actions = actions_interpolation(config, pre_action, inference_actions)
        inference_timestep = t
        inference_lock.release()
        break


def actions_interpolation(config, pre_action, actions):
    steps = np.concatenate((np.array(config.arm_steps_length), np.array(config.arm_steps_length)), axis=0)

    result = [pre_action]
    post_action = denormalize(actions[0])

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

    while len(result) < config["chunk_size"] + 1:
        result.append(result[-1])
    result = np.array(result)[1 : config["chunk_size"] + 1]  # Updated from args.chunk_size to config["chunk_size"]
    # print("actions_interpolation2:", result.shape, result[:, 7:])
    result = normalize(result)
    result = result[np.newaxis, :]
    return result

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    ros_operator = RosOperator(config)

    model = ACT(config)
    # 源代码中model的is_pad_head层的weight和bias没有用，没有key和value。同样的还有input_proj_next_action的weight和bias，注意对照记录确认一下这个的影响
    model.deserialize(torch.load(config["load_model_path"]))

    model.cuda()
    model.eval()

    ros_operator.setup_puppet_arm()

    with torch.inference_mode():
        while True and not rospy.is_shutdown():
            # 每个回合的步数
            t = 0
            max_t = 0

            rate = rospy.Rate(config["publish_rate"])

            if config["temporal_agg"]:
                all_time_actions = np.zeros([config["max_publish_step"], config["max_publish_step"] + config["chunk_size"], config["state_dimension"]])

            while t < config["max_publish_step"] and not rospy.is_shutdown():
                if t >= max_t:
                    pre_action = action
                    inference_thread = threading.Thread(target=inference_process, args=(config, ros_operator, model, t, pre_action))
                    inference_thread.start()
                    inference_thread.join()

                    inference_lock.acquire()
                    if inference_actions is not None:
                        inference_thread = None
                        all_actions = inference_actions
                        inference_actions = None
                        max_t = t + config["pos_lookahead_step"]
                        if config["temporal_agg"]:
                            all_time_actions[[t], t : t + config["chunk_size"]] = all_actions
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
                    if config["pos_lookahead_step"] != 0:
                        raw_action = all_actions[:, t % config["pos_lookahead_step"]]
                    else:
                        raw_action = all_actions[:, t % config["chunk_size"]]

                action = denormalize(raw_action[0])
                left_action = action[:7]  # 取7维度
                right_action = action[7:14]
                left_action[-1:] *= 13
                right_action[-1:] *= 13
                # left_action[-1:] = 0.5
                ros_operator.puppet_arm_publish(left_action, right_action)  # puppet_arm_publish_continuous_thread

                t += 1
                # end_time = time.time()
                # print("publish: ", t)
                # print("time:", end_time - start_time)
                # print("left_action:", left_action)
                # print("right_action:", right_action)
                rate.sleep()
