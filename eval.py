import json
import torch
from tqdm import tqdm
import yaml

from dataloader import get_dataloader
from model import ACT

with open("./config/evaluate.yaml", "r") as f:
    config = yaml.safe_load(f)

train_dataloader, valid_dataloader = get_dataloader(config)

print("dataset loaded")

model = ACT(config)
model.deserialize(torch.load(config["load_model_path"]))

print("model loaded")

model.cuda()

train_result = []
valid_result = []

with torch.inference_mode():
    for current_joint, images, qpos_chunk in tqdm(train_dataloader):
        current_joint, images, qpos_chunk = current_joint.cuda(), images.cuda(), qpos_chunk.cuda()
        _, predict_qpos_chunk = model(current_joint, images, qpos_chunk)

        for i in range(len(current_joint)):
            train_result.append(
                {
                    "current_joint": current_joint[i, :].cpu().numpy().tolist(),
                    "qpos_0": qpos_chunk[i, 0, :].cpu().numpy().tolist(),
                    "predict_qpos_0": predict_qpos_chunk[i, 0, :].cpu().numpy().tolist(),
                    "qpos_1": qpos_chunk[i, 1, :].cpu().numpy().tolist(),
                    "predict_qpos_1": predict_qpos_chunk[i, 1, :].cpu().numpy().tolist(),
                    "qpos_5": qpos_chunk[i, 5, :].cpu().numpy().tolist(),
                    "predict_qpos_5": predict_qpos_chunk[i, 5, :].cpu().numpy().tolist(),
                    "qpos_10": qpos_chunk[i, 10, :].cpu().numpy().tolist(),
                    "predict_qpos_10": predict_qpos_chunk[i, 10, :].cpu().numpy().tolist(),
                }
            )

    for current_joint, images, qpos_chunk in tqdm(valid_dataloader):
        current_joint, images, qpos_chunk = current_joint.cuda(), images.cuda(), qpos_chunk.cuda()
        _, predict_qpos_chunk = model(current_joint, images, qpos_chunk)

        for i in range(len(current_joint)):
            valid_result.append(
                {
                    "current_joint": current_joint[i, :].cpu().numpy().tolist(),
                    "qpos_0": qpos_chunk[i, 0, :].cpu().numpy().tolist(),
                    "predict_qpos_0": predict_qpos_chunk[i, 0, :].cpu().numpy().tolist(),
                    "qpos_1": qpos_chunk[i, 1, :].cpu().numpy().tolist(),
                    "predict_qpos_1": predict_qpos_chunk[i, 1, :].cpu().numpy().tolist(),
                    "qpos_5": qpos_chunk[i, 5, :].cpu().numpy().tolist(),
                    "predict_qpos_5": predict_qpos_chunk[i, 5, :].cpu().numpy().tolist(),
                    "qpos_10": qpos_chunk[i, 10, :].cpu().numpy().tolist(),
                    "predict_qpos_10": predict_qpos_chunk[i, 10, :].cpu().numpy().tolist(),
                }
            )

with open("train_result.json", "w") as f:
    json.dump(train_result, f)

with open("valid_result.json", "w") as f:
    json.dump(valid_result, f)
