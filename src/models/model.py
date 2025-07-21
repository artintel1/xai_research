import torch
import torch.nn as nn
import torchvision.models.video as video_models

class MoEModel(nn.Module):
    """
    Mixture-of-Experts (MoE) model for video action recognition with temporal sampling experts.

    This model uses a shared 3D CNN backbone for multiple temporal sampling rates
    (experts), a lightweight gating network to combine their outputs, and a
    classification head.
    """
    def __init__(self, num_classes: int, temporal_sampling_rates: tuple = (1, 2, 4, 8)):
        """
        Args:
            num_classes (int): The number of action classes.
            temporal_sampling_rates (tuple): A tuple of integers representing
                                             the 'drop' rates for each temporal expert.
        """
        super().__init__()
        self.temporal_sampling_rates = temporal_sampling_rates
        self.num_experts = len(temporal_sampling_rates)
        self.feature_size = 512  # Output feature size of the R3D_18 backbone

        # 1. Shared-Weight 3D CNN Backbone (Our "Experts")
        # We use r3d_18 and remove its original classification head.
        self.backbone = video_models.r3d_18(weights=video_models.R3D_18_Weights.DEFAULT)
        # The original R3D_18 has a final FC layer `fc` which we need to replace
        # or effectively ignore. We will use the output before this layer.
        self.backbone.fc = nn.Identity() # Replace the classification head with an identity layer

        # 2. Lightweight Gating Network
        # It takes features from one expert (e.g., the base clip_drop_1) to predict routing weights.
        self.gating_network = nn.Sequential(
            nn.Linear(self.feature_size, self.feature_size // 2),
            nn.ReLU(),
            nn.Linear(self.feature_size // 2, self.num_experts)
        )

        # 3. Aggregation and Classification Head
        self.classifier_head = nn.Linear(self.feature_size, num_classes)

    def forward(self, expert_clips: dict):
        """
        Forward pass for the MoE model.

        Args:
            expert_clips (dict): A dictionary where keys are 'clip_drop_X' (e.g., 'clip_drop_1')
                                 and values are torch.Tensors representing the video clips
                                 sampled at different rates.
                                 Shape of each clip: (batch_size, C, F, H, W).

        Returns:
            tuple:
                - torch.Tensor: Logits for classification (batch_size, num_classes).
                - torch.Tensor: Gating weights (batch_size, num_experts).
        """
        expert_features = []

        # Pass each expert clip through the shared backbone
        for rate in self.temporal_sampling_rates:
            key = f'clip_drop_{rate}'
            if key not in expert_clips:
                raise ValueError(f"Missing expected expert clip: {key}")

            clip = expert_clips[key]
            # Ensure the clip has 5 dimensions: (batch, channels, frames, height, width)
            if clip.dim() == 4: # If it's (batch, C, H, W) for a single frame, unsqueeze F
                clip = clip.unsqueeze(2) # Add a frame dimension of 1

            # The R3D backbone expects (batch, C, F, H, W)
            features = self.backbone(clip)
            expert_features.append(features)

        # Stack expert features: list of (batch_size, feature_size) -> (batch_size, num_experts, feature_size)
        expert_features_tensor = torch.stack(expert_features, dim=1)

        # Gating network input: Use features from the 'clip_drop_1' expert (highest resolution)
        # as a representative input for the gating decision.
        gating_input = expert_features_tensor[:, self.temporal_sampling_rates.index(1), :]

        # Compute routing weights
        raw_gating_weights = self.gating_network(gating_input)
        gating_weights = torch.softmax(raw_gating_weights, dim=-1) # (batch_size, num_experts)

        # Aggregate expert features using gating weights (weighted average)
        # expert_features_tensor: (B, E, D) where E is num_experts, D is feature_size
        # gating_weights: (B, E)
        # Unsqueeze gating_weights to (B, E, 1) for broadcasting during multiplication
        aggregated_features = torch.sum(expert_features_tensor * gating_weights.unsqueeze(-1), dim=1)

        # Final classification
        logits = self.classifier_head(aggregated_features)

        return logits, gating_weights
