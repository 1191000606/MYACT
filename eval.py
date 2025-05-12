import torch
import wandb

def evaluate_loss(model, valid_dataloader):
    with torch.inference_mode():
        model.eval()

        valid_loss = 0.0

        for current_joint, images, qpos_chunk in valid_dataloader:
            current_joint, images, qpos_chunk = current_joint.cuda(), images.cuda(), qpos_chunk.cuda()
            loss, _ = model(current_joint, images, qpos_chunk)

            valid_loss += loss.item()

            wandb.log({"batch_valid_loss": loss.item()})

        return valid_loss / len(valid_dataloader)
