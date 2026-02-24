import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.utils.tensorboard as tb

from .models import load_model, save_model
from .metrics import PlannerMetric
from .datasets.road_dataset import load_data


def weighted_l1_loss(preds, targets, mask):
    mask = mask.float().unsqueeze(-1)
    l1_diff = torch.abs(preds - targets) * mask
    valid_count = mask[..., 0].sum() + 1e-6
    longitudinal_loss = l1_diff[..., 0].sum() / valid_count
    lateral_loss = l1_diff[..., 1].sum() / valid_count
    return longitudinal_loss + lateral_loss


def train_step(model, train_data, optimizer, device, model_name, **kwargs):
    model.train()
    total_loss = 0
    for batch in train_data:
        # Load batch data
        track_left = batch["track_left"].to(device)
        track_right = batch["track_right"].to(device)
        waypoints = batch["waypoints"].to(device)
        waypoints_mask = batch["waypoints_mask"].to(device)
        image = batch["image"].to(device)

        optimizer.zero_grad()
        preds = model(image, **kwargs) if model_name == "cnn_planner" else model(track_left, track_right, **kwargs)
        loss = weighted_l1_loss(preds, waypoints, waypoints_mask)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_data)


def validation_step(model, val_data, metrics, device, model_name, **kwargs):
    model.eval()
    with torch.inference_mode():
        for batch in val_data:
            track_left = batch["track_left"].to(device)
            track_right = batch["track_right"].to(device)
            waypoints = batch["waypoints"].to(device)
            waypoints_mask = batch["waypoints_mask"].to(device)
            image = batch["image"].to(device)

            preds = model(image, **kwargs) if model_name == "cnn_planner" else model(track_left, track_right, **kwargs)
            metrics.add(preds, waypoints, waypoints_mask)


def log_metrics(logger, metrics, epoch):
    val_metrics = metrics.compute()
    for key, value in val_metrics.items():
        logger.add_scalar(key, value, epoch)
    return val_metrics


def train(exp_dir="logs", model_name="linear", num_epoch=50, lr=1e-3, batch_size=128, seed=2024, **kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    model = load_model(model_name, **kwargs).to(device)
    train_data = load_data("drive_data/train", shuffle=True, batch_size=batch_size, num_workers=4)
    val_data = load_data("drive_data/val", shuffle=False, batch_size=batch_size, num_workers=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    metrics = PlannerMetric()

    for epoch in range(num_epoch):
        metrics.reset()
        train_loss = train_step(model, train_data, optimizer, device, model_name, **kwargs)
        validation_step(model, val_data, metrics, device, model_name, **kwargs)
        val_metrics = log_metrics(logger, metrics, epoch)

        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d}/{num_epoch:2d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val L1 Error: {val_metrics['l1_error']:.4f} | "
                f"Longitudinal Error: {val_metrics['longitudinal_error']:.4f} | "
                f"Lateral Error: {val_metrics['lateral_error']:.4f} | "
                f"Samples: {val_metrics['num_samples']:.4f}"
            )

    save_model(model)
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2024)
    train(**vars(parser.parse_args()))
