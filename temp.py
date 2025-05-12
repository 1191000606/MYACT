import os
import h5py
import numpy as np
import torch
import yaml

with open('./config/train.yaml', 'r') as f:
    config = yaml.safe_load(f)

all_qpos_data = []

for episode_index in range(config["episode_num"]):
    dataset_path = os.path.join(config["dataset_dir"], config["task_name"], f"episode_{episode_index}.hdf5")

    with h5py.File(dataset_path, "r") as root:
        qpos = root["/observations/qpos"][()]
        all_qpos_data.append(torch.from_numpy(qpos))

all_qpos_flatten = torch.cat(all_qpos_data, dim=0)

qpos_mean = all_qpos_flatten.mean(dim=0)

qpos_std = all_qpos_flatten.std(dim=0)
qpos_std = torch.clip(qpos_std, 1e-2, np.inf)

qpos_mean, qpos_std = qpos_mean.numpy(), qpos_std.numpy()

print(qpos_mean)
print(qpos_std)