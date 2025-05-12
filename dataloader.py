import numpy as np
import torch

from dataset import EpisodicDataset

from torch.utils.data import DataLoader


def get_dataloader(config):
    if config["load_preprocess_dataset"]:
        dataset_dir = f"{config['dataset_dir']}/{config['task_name']}/"

        train_dataset = torch.load(dataset_dir + "train_dataset.pt", weights_only=False)
        valid_dataset = torch.load(dataset_dir + "valid_dataset.pt", weights_only=False)
    else:
        train_ratio = 0.8

        shuffled_ids = np.random.permutation(config["episode_num"])

        train_ids = shuffled_ids[: int(train_ratio * config["episode_num"])]
        valid_ids = shuffled_ids[int(train_ratio * config["episode_num"]) :]

        train_dataset = EpisodicDataset(config, train_ids)
        valid_dataset = EpisodicDataset(config, valid_ids)

        torch.save(train_dataset, "train_dataset.pt")
        torch.save(valid_dataset, "valid_dataset.pt")

    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    valid_dataloader = DataLoader(valid_dataset, batch_size=config["batch_size"], shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, valid_dataloader
