import numpy as np
import torch
from eval import evaluate_loss
from tqdm import tqdm
import time
from model import ACT
import wandb


def train_model(config, train_dataloader, valid_dataloader):
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    model = ACT(config)
    if config["load_model_path"] is not None:
        print(f"Loading model from {config['load_model_path']}")
        model.deserialize(torch.load(config["load_model_path"]))
    model.cuda()

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": config["backbone_learning_rate"],
        },
    ]

    optimizer = torch.optim.AdamW(param_dicts, lr=config["learning_rate"], weight_decay=config["weight_decay"])

    best_valid_loss = float("inf")
    if config["load_model_path"] is not None:
        best_valid_loss = evaluate_loss(model, valid_dataloader)
        print(f"Initial validation loss: {best_valid_loss:.4f}")

    for epoch in range(config["epoch_num"]):
        start = time.time()

        train_loss = train(model, train_dataloader, optimizer)
        valid_loss = evaluate_loss(model, valid_dataloader)

        end = time.time()
        duration = end - start

        print("-" * 50)
        print(f"Epoch {epoch:2d} | Time {duration:5.4f} sec | Train Loss {train_loss:5.4f} | valid Loss {valid_loss:5.4f}")
        print("-" * 50)

        wandb.log({"epoch_train_loss": train_loss, "epoch_valid_loss": valid_loss})

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            print(f"New best model found at epoch {epoch} with validation loss {valid_loss:.4f}")
            save_path = f"{config['checkpoint_dir']}/{config['datetime']}/best_model.pth"
            torch.save(model.serialize(), save_path)
            print(f"Best model saved at {save_path}")

        save_path = f"{config['checkpoint_dir']}/{config['datetime']}/model_epoch_{epoch}.pth"
        torch.save(model.serialize(), save_path)
        print(f"Model saved at epoch {epoch}")

    return model


def train(model, train_dataloader, optimizer):
    model.train()
    optimizer.zero_grad()

    train_loss = 0.0

    for current_joint, images, qpos_chunk in tqdm(train_dataloader):
        current_joint, images, qpos_chunk = current_joint.cuda(), images.cuda(), qpos_chunk.cuda()
        
        loss, _ = model(current_joint, images, qpos_chunk)

        wandb.log({"batch_train_loss": loss.item()})
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()
    
    return train_loss / len(train_dataloader)
