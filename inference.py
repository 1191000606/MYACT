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

def inference(model, ros_operator):
    frame = ros_operator.get_frame()
    
    img_front, img_left, img_right, puppet_arm_left, puppet_arm_right = frame

    qpos = np.concatenate((np.array(puppet_arm_left.position), np.array(puppet_arm_right.position)), axis=0)
    qpos = normalize(qpos)
    qpos = torch.from_numpy(qpos).float().cuda()

    curr_images = []
    for image in [img_front, img_left, img_right]:
        curr_image = rearrange(image, "h w c -> c h w")
        curr_images.append(curr_image)
    
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda()

    chunk_actions = model(qpos.unsqueeze(0), curr_image.unsqueeze(0))[0]

    return denormalize(chunk_actions.cpu().detach().numpy())

if __name__ == "__main__":
    with open("./config/inference.yaml", "r") as f:
        config = yaml.safe_load(f)

    ros_operator = RosOperator(config)

    model = ACT(config)
    # 源代码中model的is_pad_head层的weight和bias没有用，没有key和value。同样的还有input_proj_next_action的weight和bias，注意对照记录确认一下这个的影响
    model.deserialize(torch.load(config["load_model_path"]))

    model.cuda()

    ros_operator.setup_puppet_arm()

    with torch.inference_mode():
        while True and not rospy.is_shutdown():
            rate = rospy.Rate(config["publish_rate"])

            aggregate_action_array = np.zeros((config["episode_length"] + config["chunk_size"], config["state_dimension"]), dtype=np.float32)

            current_time = 0

            while current_time < config["episode_length"]:
                if current_time % config["inference_interval"] == 0:
                    chunk_actions = inference(model, ros_operator)

                    aggregate_action_array[current_time:current_time + config["chunk_size"], :] = chunk_actions * config["aggregate_decay"] + aggregate_action_array[current_time:current_time + config["chunk_size"], :] * (1 - config["aggregate_decay"])
                
                ros_operator.puppet_arm_publish_continuous(aggregate_action_array[current_time][:7], aggregate_action_array[current_time][7:])

                current_time += 1
                
                rate.sleep()
                







