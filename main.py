# import debugpy
# debugpy.connect(("192.168.1.50", 16789))
# debugpy.wait_for_client()
# debugpy.breakpoint()

import datetime
import yaml
import wandb
import os

from dataloader import get_dataloader

from train import train_model


if __name__ == '__main__':
    with open('./config/train.yaml', 'r') as f:
        config = yaml.safe_load(f)
        config["datetime"] = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        os.makedirs(f"{config['checkpoint_dir']}/{config['datetime']}", exist_ok=True)

    wandb.init(project="act_training", config=config)

    train_dataloader, valid_dataloader = get_dataloader(config)
    
    model = train_model(config, train_dataloader, valid_dataloader) 

# Todo: 目前机械臂还是用的关节空间的坐标系，后续需要换成末端执行器的坐标系
# Todo: 对图像数据的处理是不是也漏了