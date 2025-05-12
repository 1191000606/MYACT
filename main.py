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
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        config["datetime"] = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        os.makedirs(f"{config['checkpoint_dir']}/{config['datetime']}", exist_ok=True)

    wandb.init(project="act_training", config=config)

    train_dataloader, valid_dataloader = get_dataloader(config)
    
    model = train_model(config, train_dataloader, valid_dataloader) 
