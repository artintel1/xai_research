import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import os

# Assuming these are available in the project structure
from src.data.dataset import VideoDataset
from src.models.model import MoEModel

def calculate_load_balancing_loss(gating_weights, expert_outputs, num_experts):
    """
    Calculates the auxiliary load balancing loss for MoE.
    Encourages experts to be used somewhat evenly.

    Args:
        gating_weights (torch.Tensor): Tensor of gating weights (batch_size, num_experts).
        expert_outputs (torch.Tensor): Features produced by experts (batch_size, num_experts, feature_size).
        num_experts (int): Total number of experts.

    Returns:
        torch.Tensor: The scalar load balancing loss.
    """
    # Sum of gating weights per expert across the batch: (num_experts,)
    # This represents how much each expert is "activated"
    expert_load = gating_weights.sum(dim=0) # (num_experts,)

    # Mean of expert outputs across the batch (not directly used for this type of load balancing)
    # This is more for a 'importance' balancing, not just usage balancing
    # For a simple load balancing loss, we want to ensure expert_load is evenly distributed.

    # We want expert_load to be close to batch_size / num_experts for each expert.
    # A common formulation involves the variance or simply
    # (sum_i p_i x_i) - (sum_i p_i)(sum_i x_i) or similar, where p_i are probabilities.

    # A simpler approach is to encourage the variance of expert_load to be low.
    # Or, as in some MoE papers, promote (sum_i g_i) * (sum_i (g_i)^2)

    # Let's use a simpler version: penalize if sum(p_i)^2 != N * sum(p_i * x_i)
    # A more standard one from actual MoE papers:
    # (sum_i expert_gating_sum_for_batch[i]) * (sum_i expert_output_mean_for_batch[i])

    # Let's use the method from "Outrageously Large Neural Networks" (Shazeer et al. 2017)
    # Loss = sum_{i=1 to N} (P_i * M_i)
    # Where P_i is the fraction of total samples routed to expert i, and M_i is the mean output magnitude.
    # Or, a simpler formulation: (sum_i mean_gating_weights_for_expert_i) * (sum_i mean_expert_feature_norm_for_expert_i)

    # Let's stick to the simpler one that encourages equal usage:
    # A common way is to make sure the mean gating weight for each expert is similar.
    # This can be approximated by minimizing the covariance between dispatching and expert usage.

    # Simpler version: variance of expert_load
    load_balancing_loss = torch.var(expert_load)

    # Another common form:
    # From "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding"
    # L_balance = Sum_i (P_i * F_i) where P_i = sum_k gating_weights[k,i] and F_i = sum_k I(expert_k is top1_for_input_k)
    # This might be too complex for simple implementation.

    # A more direct load balancing from "Outrageously Large Neural Networks":
    # Loss = sum_{experts} sum_{batch_elements} gating_weight[expert] * (expert_output * 1.0 / (number_of_times_expert_is_chosen))
    # This implies knowledge of which expert is chosen, which is for sparse MoE.

    # For dense MoE where all experts are always "chosen" to some degree,
    # a simpler balancing loss could be to penalize the squared difference from uniform distribution.
    # Or, directly from the paper: `loss += (load * importance).sum()`
    # where load is sum(gating_weight) for expert, and importance is expert_output_mean_norm.

    # Let's use the simplest, most direct one that encourages all experts to have non-zero average weight:
    # sum (p_i)^2 where p_i is average probability for expert i
    avg_gating_weights = gating_weights.mean(dim=0) # (num_experts,)
    load_balancing_loss = (avg_gating_weights * avg_gating_weights).sum() * num_experts # Penalize small sums
    # Or, simply minimize the entropy of avg_gating_weights
    # Or, maximize the entropy of avg_gating_weights, or make them uniform.

    # For this implementation, let's target equal average usage per expert.
    # Penalize the deviation of each expert's average gate value from the overall average.
    # (batch_size * num_experts) * Sum_i (Avg(g_i) - 1/num_experts)^2
    # The sum of all gating weights for a batch / batch size should be 1.0.
    # So avg_gating_weights should be close to 1.0 / num_experts for each expert.
    # We want to minimize the variance of `avg_gating_weights`.
    load_balancing_loss = avg_gating_weights.var()

    return load_balancing_loss


def train(model, dataloader, criterion_cls, criterion_lb, optimizer, device, lambda_lb):
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_lb_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for expert_clips, labels in tqdm(dataloader, desc="Training"):
        expert_clips_on_device = {k: v.to(device) for k, v in expert_clips.items()}
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        logits, gating_weights = model(expert_clips_on_device)

        # Calculate classification loss
        cls_loss = criterion_cls(logits, labels)

        # Calculate load balancing loss
        # The expert_outputs (features) are needed for a more complete balancing,
        # but for simple usage balancing, only gating_weights are needed.
        # Here, expert_outputs are implicitly the pre-softmax outputs from the gating network.
        # For simplicity, we'll use gating_weights only for the load balancing calculation.
        lb_loss = calculate_load_balancing_loss(gating_weights, None, model.num_experts)

        # Total loss
        loss = cls_loss + lambda_lb * lb_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_lb_loss += lb_loss.item()

        _, predicted = torch.max(logits, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    avg_cls_loss = total_cls_loss / len(dataloader)
    avg_lb_loss = total_lb_loss / len(dataloader)
    accuracy = correct_predictions / total_samples
    return avg_loss, avg_cls_loss, avg_lb_loss, accuracy

def validate(model, dataloader, criterion_cls, device):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for expert_clips, labels in tqdm(dataloader, desc="Validation"):
            expert_clips_on_device = {k: v.to(device) for k, v in expert_clips.items()}
            labels = labels.to(device)

            logits, _ = model(expert_clips_on_device) # Gating weights not needed for validation loss

            loss = criterion_cls(logits, labels)
            total_loss += loss.item()

            _, predicted = torch.max(logits, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy

def main():
    parser = argparse.ArgumentParser(description="MoE Temporal Sampling Action Recognition Training")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to the root data directory (e.g., data/KARD_video_organized)")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Name of the dataset (e.g., KARD, UCF-101, HMDB-51)")
    parser.add_argument("--num_frames", type=int, default=16,
                        help="Number of frames per expert clip.")
    parser.add_argument("--sampling_rates", type=int, nargs='+', default=[1, 2, 4, 8],
                        help="Temporal sampling rates for experts (e.g., 1 2 4 8).")
    parser.add_argument("--height", type=int, default=224, help="Frame height.")
    parser.add_argument("--width", type=int, default=224, help="Frame width.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--num_epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--lambda_lb", type=float, default=0.01,
                        help="Coefficient for the load balancing loss.")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay (L2 penalty) for the optimizer.")
    parser.add_argument("--dropout_prob", type=float, default=0.5,
                        help="Dropout probability for the classifier head.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers.")
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                        help="Directory to save model checkpoints.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training (cuda or cpu).")
    args = parser.parse_args()

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"Using device: {args.device}")

    # Data loading
    train_data_path = os.path.join(args.data_dir, args.dataset_name, 'train')
    val_data_path = os.path.join(args.data_dir, args.dataset_name, 'val') # Assuming a 'val' split

    train_dataset = VideoDataset(
        data_dir=train_data_path,
        num_frames=args.num_frames,
        temporal_sampling_rates=tuple(args.sampling_rates),
        height=args.height, width=args.width, is_train=True
    )
    val_dataset = VideoDataset(
        data_dir=val_data_path,
        num_frames=args.num_frames,
        temporal_sampling_rates=tuple(args.sampling_rates),
        height=args.height, width=args.width, is_train=False
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # Determine number of classes
    num_classes = len(train_dataset.class_to_idx)
    print(f"Number of classes: {num_classes}")

    # Model, Loss, Optimizer
    model = MoEModel(
        num_classes=num_classes,
        temporal_sampling_rates=tuple(args.sampling_rates),
        dropout_prob=args.dropout_prob
    ).to(args.device)
    criterion_cls = nn.CrossEntropyLoss()
    # For load balancing loss, we defined a helper function
    criterion_lb = calculate_load_balancing_loss # Using the helper function directly
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_val_accuracy = -1.0
    best_model_path = os.path.join(args.save_dir, "best_model.pth")
    final_model_path = os.path.join(args.save_dir, "final_model.pth")


    print("Starting training...")
    for epoch in range(1, args.num_epochs + 1):
        train_loss, train_cls_loss, train_lb_loss, train_accuracy = train(
            model, train_loader, criterion_cls, criterion_lb, optimizer, args.device, args.lambda_lb
        )
        val_loss, val_accuracy = validate(
            model, val_loader, criterion_cls, args.device
        )

        print(f"Epoch {epoch}/{args.num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f} (Cls: {train_cls_loss:.4f}, LB: {train_lb_loss:.4f}) | Train Acc: {train_accuracy:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}")

        # Save best model based on validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f"  Saved best model with Val Acc: {best_val_accuracy:.4f} to {best_model_path}")

    # Save the final model
    torch.save(model.state_dict(), final_model_path)
    print(f"Training complete! Final model saved to {final_model_path}")

if __name__ == "__main__":
    main()
