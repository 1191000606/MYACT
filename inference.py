import torch
import yaml
from aloha import RosOperator
from model import ACT


if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    ros_operator = RosOperator(config)

    global inference_lock
    global inference_actions
    global inference_timestep
    global inference_thread

    model = ACT(config)
    # 源代码中model的is_pad_head层的weight和bias没有用，没有key和value。同样的还有input_proj_next_action的weight和bias，注意对照记录确认一下这个的影响
    model.deserialize(torch.load(config["load_model_path"]))

    model.cuda()
    model.eval()

    