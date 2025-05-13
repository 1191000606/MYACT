import os
import h5py
import numpy as np
import torch


def qpos_normalization(config):
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

    return qpos_mean.numpy(), qpos_std.numpy()


class EpisodicDataset(torch.utils.data.Dataset):

    def __init__(self, config, ids):
        super(EpisodicDataset).__init__()

        self.qpos_mean, self.qpos_std = qpos_normalization(config)

        qpos_list = []
        images_list = []
        for id in ids:
            dataset_path = os.path.join(config["dataset_dir"], config["task_name"], f"episode_{id}.hdf5")

            with h5py.File(dataset_path, "r") as file:
                qpos = file["/observations/qpos"][()]
                qpos = self.normalize(qpos)

                images = []
                for camera_name in config["camera_names"]:
                    images.append(file[f"/observations/images/{camera_name}"][()])

                images = np.stack(images, axis=1)
                images = torch.from_numpy(images)
                images = torch.einsum("l k h w c -> l k c h w", images)  # 这一步可以直接使用from einops import rearrange
                images = images / 255.0

                qpos_list.append(qpos)
                images_list.append(images)

        self.current_joint_list = []
        self.images_list = []
        self.qos_chunk_list = []
        for qpos, images in zip(qpos_list, images_list):
            for i in range(0, len(qpos) - config["chunk_size"] - 1):
                self.current_joint_list.append(qpos[i])
                self.images_list.append(images[i])
                self.qos_chunk_list.append(qpos[i + 1 : i + 1 + config["chunk_size"]])

    def normalize(self, qpos):
        return (qpos - self.qpos_mean) / self.qpos_std

    def denormalize(self, qpos):
        return qpos * self.qpos_std + self.qpos_mean

    def __len__(self):
        return len(self.qos_chunk_list)

    def __getitem__(self, index):
        return self.current_joint_list[index], self.images_list[index], self.qos_chunk_list[index]
