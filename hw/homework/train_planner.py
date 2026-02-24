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


def weighted_l1_loss(preds, targets, mask, model_name="mlp_planner"):
    mask = mask.float().unsqueeze(-1)
    l1_diff = torch.abs(preds - targets) * mask
    valid_count = mask[..., 0].sum() + 1e-6
    longitudinal_loss = l1_diff[..., 0].sum() / valid_count
    lateral_loss = l1_diff[..., 1].sum() / valid_count
    
    # Weight lateral error more heavily since it's the bottleneck
    # CNN needs more aggressive lateral error reduction (threshold 0.45 vs 0.6)
    if model_name == "cnn_planner":
        lateral_weight = 3.0  # More aggressive for CNN
    else:
        lateral_weight = 2.5  # Strong focus for MLP/Transformer
    
    return longitudinal_loss + lateral_weight * lateral_loss


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
        loss = weighted_l1_loss(preds, waypoints, waypoints_mask, model_name)
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
    train_data = load_data("drive_data/train", shuffle=True, batch_size=batch_size, num_workers=2)
    val_data = load_data("drive_data/val", shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Add learning rate scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch)
    metrics = PlannerMetric()
    
    # Track best model based on lateral error (most important metric)
    best_lateral_error = float('inf')
    best_epoch = 0

    for epoch in range(num_epoch):
        metrics.reset()
        train_loss = train_step(model, train_data, optimizer, device, model_name, **kwargs)
        validation_step(model, val_data, metrics, device, model_name, **kwargs)
        val_metrics = log_metrics(logger, metrics, epoch)
        scheduler.step()
        
        # Save best model based on lateral error (most critical metric)
        current_lateral_error = val_metrics['lateral_error']
        if current_lateral_error < best_lateral_error:
            best_lateral_error = current_lateral_error
            best_epoch = epoch + 1
            # Save best checkpoint to log directory
            best_checkpoint_path = log_dir / f"{model_name}_best.th"
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"  → New best model! Lateral error: {best_lateral_error:.4f}")

        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d}/{num_epoch:2d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val L1 Error: {val_metrics['l1_error']:.4f} | "
                f"Longitudinal Error: {val_metrics['longitudinal_error']:.4f} | "
                f"Lateral Error: {val_metrics['lateral_error']:.4f} | "
                f"Best Lateral: {best_lateral_error:.4f} (epoch {best_epoch})"
            )

    # Load and save the best model (not the final epoch model)
    print(f"\nTraining complete! Best lateral error: {best_lateral_error:.4f} at epoch {best_epoch}")
    
    # Load best checkpoint and save to homework directory
    best_checkpoint_path = log_dir / f"{model_name}_best.th"
    if best_checkpoint_path.exists():
        model.load_state_dict(torch.load(best_checkpoint_path, map_location=device))
        save_model(model)  # Save best model to homework/
        print(f"✓ Best model loaded and saved to homework/{model_name}.th")
    else:
        # Fallback: if no best model, save current model
        save_model(model)
        print(f"⚠ Warning: Best checkpoint not found, saved final epoch model")
    
    # Also save final training model to log directory for comparison
    torch.save(model.state_dict(), log_dir / f"{model_name}_final.th")
    print(f"✓ Training log saved to: {log_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)
    train(**vars(parser.parse_args()))
