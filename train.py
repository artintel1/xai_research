import os
import argparse
import yaml
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Assuming your project structure is project_xai/train.py, project_xai/src/...
from src.data.dataset import VideoDataset
from src.models.model import MoEModel


def calculate_load_balancing_loss(gating_weights, num_experts):
    """
    Calculates the auxiliary load balancing loss for MoE.
    Encourages experts to be used somewhat evenly.

    Args:
        gating_weights (torch.Tensor): Tensor of gating weights (batch_size, num_experts).
        num_experts (int): Total number of experts.

    Returns:
        torch.Tensor: The scalar load balancing loss (variance of average gate values).
    """
    if gating_weights is None or gating_weights.numel() == 0:
        return torch.tensor(0.0, device=gating_weights.device)
    # Average gating weight per expert across the batch
    avg_gating_weights = gating_weights.mean(dim=0)
    # Penalize the variance of these average gating weights to encourage uniform usage.
    load_balancing_loss = avg_gating_weights.var()
    return load_balancing_loss


def train(model, dataloader, criterion_cls, criterion_lb_func, optimizer, device, cfg):
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_spatial_lb_loss = 0.0
    total_temporal_lb_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for expert_clips_nested, labels in tqdm(dataloader, desc="Training"):
        labels = labels.to(device)
        expert_clips_on_device = {}
        for res_key, temporal_clips_dict in expert_clips_nested.items():
            expert_clips_on_device[res_key] = {k: v.to(device) for k, v in temporal_clips_dict.items()}

        optimizer.zero_grad()

        logits, spatial_gating_weights, all_temporal_gating_weights = model(expert_clips_on_device)

        cls_loss = criterion_cls(logits, labels)
        spatial_lb_loss = criterion_lb_func(spatial_gating_weights, model.num_spatial_experts)

        current_temporal_lb_loss_sum = 0.0
        for res_key, temp_gating_weights in all_temporal_gating_weights.items():
            current_temporal_lb_loss_sum += criterion_lb_func(temp_gating_weights, model.num_temporal_experts)

        avg_temporal_lb_loss = current_temporal_lb_loss_sum / len(all_temporal_gating_weights) if all_temporal_gating_weights else torch.tensor(0.0, device=device)

        loss = cls_loss + cfg['moe']['lambda_lb_spatial'] * spatial_lb_loss + cfg['moe']['lambda_lb_temporal'] * avg_temporal_lb_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_spatial_lb_loss += spatial_lb_loss.item()
        total_temporal_lb_loss += avg_temporal_lb_loss.item()

        _, predicted = torch.max(logits, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    avg_cls_loss = total_cls_loss / len(dataloader)
    avg_spatial_lb_loss = total_spatial_lb_loss / len(dataloader)
    avg_temporal_lb_loss = total_temporal_lb_loss / len(dataloader)
    accuracy = correct_predictions / total_samples
    return avg_loss, avg_cls_loss, avg_spatial_lb_loss, avg_temporal_lb_loss, accuracy


def validate(model, dataloader, criterion_cls, device):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for expert_clips_nested, labels in tqdm(dataloader, desc="Validation"):
            labels = labels.to(device)
            expert_clips_on_device = {}
            for res_key, temporal_clips_dict in expert_clips_nested.items():
                expert_clips_on_device[res_key] = {k: v.to(device) for k, v in temporal_clips_dict.items()}

            logits, _, _ = model(expert_clips_on_device)

            loss = criterion_cls(logits, labels)
            total_loss += loss.item()

            _, predicted = torch.max(logits, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="Two-Tier MoE Action Recognition Training")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    # Add arguments for overriding specific config values, e.g., --batch_size, --learning_rate
    # These should have default=None to easily check if they were provided.
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size in config.")
    parser.add_argument("--num_epochs", type=int, default=None, help="Override number of epochs in config.")
    parser.add_argument("--learning_rate", type=float, default=None, help="Override learning rate in config.")
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Override resume_checkpoint path in config.")

    args = parser.parse_args()

    # Load config from YAML file
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # --- Override config with command-line arguments if provided ---
    if args.num_epochs is not None:
        cfg['training']['num_epochs'] = args.num_epochs
    if args.batch_size is not None:
        cfg['training']['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        cfg['training']['learning_rate'] = args.learning_rate
    if args.resume_checkpoint is not None:
        cfg['checkpoint']['resume_checkpoint'] = args.resume_checkpoint

    # --- Setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = cfg['checkpoint']['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    print("--- Configuration ---")
    print(yaml.dump(cfg))
    print("---------------------")
    print(f"Using device: {device}")

    # --- Data Loading ---
    train_data_path = os.path.join(cfg['data']['data_dir'], cfg['data']['dataset_name'], 'train')
    val_data_path = os.path.join(cfg['data']['data_dir'], cfg['data']['dataset_name'], 'val')

    spatial_resolutions_tuples = tuple(tuple(res) for res in cfg['data']['spatial_resolutions'])

    train_dataset = VideoDataset(
        data_dir=train_data_path,
        num_frames=cfg['data']['num_frames'],
        temporal_sampling_rates=tuple(cfg['data']['temporal_sampling_rates']),
        spatial_resolutions=spatial_resolutions_tuples,
        is_train=True
    )
    val_dataset = VideoDataset(
        data_dir=val_data_path,
        num_frames=cfg['data']['num_frames'],
        temporal_sampling_rates=tuple(cfg['data']['temporal_sampling_rates']),
        spatial_resolutions=spatial_resolutions_tuples,
        is_train=False
    )

    train_loader = DataLoader(
        train_dataset, batch_size=cfg['training']['batch_size'], shuffle=True,
        num_workers=cfg['data']['num_workers'], pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg['training']['batch_size'], shuffle=False,
        num_workers=cfg['data']['num_workers'], pin_memory=True
    )

    num_classes = len(train_dataset.class_to_idx)
    print(f"Number of classes: {num_classes}")

    # --- Model, Loss, Optimizer ---
    model = MoEModel(
        num_classes=num_classes,
        temporal_sampling_rates=tuple(cfg['data']['temporal_sampling_rates']),
        spatial_resolutions=spatial_resolutions_tuples,
        dropout_prob=cfg['model']['dropout_prob']
    ).to(device)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_lb = calculate_load_balancing_loss
    optimizer = optim.Adam(model.parameters(), lr=cfg['training']['learning_rate'], weight_decay=cfg['training']['weight_decay'])

    # --- Resumption Logic ---
    start_epoch = 1
    best_val_accuracy = -1.0
    resume_checkpoint_path = cfg['checkpoint']['resume_checkpoint']

    if resume_checkpoint_path and os.path.exists(resume_checkpoint_path):
        print(f"Resuming training from checkpoint: {resume_checkpoint_path}")
        checkpoint = torch.load(resume_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_accuracy = checkpoint.get('best_val_accuracy', -1.0)
        print(f"Resumed from epoch {checkpoint['epoch']}, with best validation accuracy {best_val_accuracy:.4f}")
    elif resume_checkpoint_path:
        print(f"Warning: --resume_checkpoint path '{resume_checkpoint_path}' not found. Starting training from scratch.")

    # --- Training Loop ---
    print("Starting training...")
    for epoch in range(start_epoch, cfg['training']['num_epochs'] + 1):
        train_loss, train_cls_loss, train_spatial_lb_loss, train_temporal_lb_loss, train_accuracy = train(
            model, train_loader, criterion_cls, criterion_lb, optimizer, device, cfg
        )
        val_loss, val_accuracy = validate(model, val_loader, criterion_cls, device)

        print(f"Epoch {epoch}/{cfg['training']['num_epochs']}:")
        print(f"  Train Loss: {train_loss:.4f} (Cls: {train_cls_loss:.4f}, Spatial LB: {train_spatial_lb_loss:.4f}, Temporal LB: {train_temporal_lb_loss:.4f}) | Train Acc: {train_accuracy:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}")

        # --- Checkpoint Saving ---
        best_model_path = os.path.join(save_dir, "best_model.pth")
        latest_model_path = os.path.join(save_dir, "latest_model.pth")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f"  Saved new best model with Val Acc: {best_val_accuracy:.4f} to {best_model_path}")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_accuracy': best_val_accuracy,
            'config': cfg
        }, latest_model_path)
        print(f"  Saved latest checkpoint to {latest_model_path}")

    print("\nTraining complete!")
    print(f"Final best model saved to {best_model_path} with accuracy {best_val_accuracy:.4f}")
    print(f"Latest checkpoint available at {latest_model_path}")

if __name__ == "__main__":
    main()
