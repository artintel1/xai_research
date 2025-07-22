import torch
import torch.nn as nn
import torchvision.models.video as video_models
import torch.nn.functional as F # For dynamic resizing (e.g., interpolate)

class MoEModel(nn.Module):
    """
    Two-tiered Mixture-of-Experts (MoE) model for video action recognition,
    with experts for both spatial resolution and temporal sampling rates.
    """
    def __init__(self, num_classes: int, temporal_sampling_rates: tuple = (1, 2, 4, 8),
                 spatial_resolutions: tuple = ((224, 224), (112, 112)), dropout_prob: float = 0.5):
        """
        Args:
            num_classes (int): The number of action classes.
            temporal_sampling_rates (tuple): A tuple of integers representing
                                             the 'drop' rates for each temporal expert.
            spatial_resolutions (tuple): A tuple of (height, width) tuples for
                                         each spatial expert resolution.
            dropout_prob (float): The probability for the dropout layer.
        """
        super().__init__()
        self.temporal_sampling_rates = temporal_sampling_rates
        self.spatial_resolutions = spatial_resolutions
        self.num_temporal_experts = len(temporal_sampling_rates)
        self.num_spatial_experts = len(spatial_resolutions)
        self.feature_size = 512  # Output feature size of the R3D_18 backbone
        self.backbone_input_size = (224, 224) # R3D_18 standard input resolution

        # 1. Shared-Weight 3D CNN Backbone
        self.backbone = video_models.r3d_18(weights=video_models.R3D_18_Weights.DEFAULT)
        self.backbone.fc = nn.Identity() # Replace the classification head with an identity layer

        # 2. Temporal Gating Network (Applied per spatial resolution)
        self.temporal_gating_network = nn.Sequential(
            nn.Linear(self.feature_size, self.feature_size // 2),
            nn.ReLU(),
            nn.Linear(self.feature_size // 2, self.num_temporal_experts)
        )

        # 3. Spatial Gating Network
        # It takes features from one representative spatial resolution's aggregated temporal features.
        self.spatial_gating_network = nn.Sequential(
            nn.Linear(self.feature_size, self.feature_size // 2),
            nn.ReLU(),
            nn.Linear(self.feature_size // 2, self.num_spatial_experts)
        )

        # 4. Final Aggregation and Classification Head
        self.classifier_head = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(self.feature_size, num_classes)
        )

    def forward(self, expert_clips_nested: dict):
        """
        Forward pass for the two-tiered MoE model.

        Args:
            expert_clips_nested (dict): A nested dictionary.
                                        Keys are 'res_H' for spatial resolutions.
                                        Values are dictionaries:
                                            Keys are 'clip_drop_R' for temporal sampling rates.
                                            Values are torch.Tensors of clips (batch_size, C, F, H, W).

        Returns:
            tuple:
                - torch.Tensor: Logits for classification (batch_size, num_classes).
                - torch.Tensor: Spatial gating weights (batch_size, num_spatial_experts).
                - dict: Dictionary of temporal gating weights {res_key: (batch_size, num_temporal_experts)}.
        """
        spatial_expert_aggregated_features = []
        all_temporal_gating_weights = {}

        # Outer loop: Iterate through each spatial resolution
        for res_idx, (height, width) in enumerate(self.spatial_resolutions):
            res_key = f'res_{height}'
            temporal_clips_for_res = expert_clips_nested.get(res_key)
            if temporal_clips_for_res is None:
                raise ValueError(f"Missing expected spatial expert clips for resolution: {res_key}")

            temporal_features_for_this_res = []

            # Inner loop: Pass each temporal expert clip through the shared backbone
            for rate in self.temporal_sampling_rates:
                clip_key = f'clip_drop_{rate}'
                if clip_key not in temporal_clips_for_res:
                    raise ValueError(f"Missing expected temporal expert clip: {clip_key} for resolution {res_key}")

                clip = temporal_clips_for_res[clip_key]
                # Ensure the clip has 5 dimensions: (batch, channels, frames, height, width)
                if clip.dim() == 4: # If it's (batch, C, H, W) for a single frame, unsqueeze F
                    clip = clip.unsqueeze(2) # Add a frame dimension of 1

                # Dynamically resize the clip to the backbone's expected input size (e.g., 224x224)
                # This applies F.interpolate on H,W dimensions for each frame in the clip.
                if (height, width) != self.backbone_input_size:
                    batch_size, C, F_val, H_val, W_val = clip.shape
                    # Reshape to (B*F, C, H, W) for F.interpolate which expects 4D input for spatial resize
                    clip_reshaped = clip.permute(0, 2, 1, 3, 4).reshape(batch_size * F_val, C, H_val, W_val)
                    resized_clip_reshaped = F.interpolate(clip_reshaped, size=self.backbone_input_size, mode='bilinear', align_corners=False)
                    # Reshape back to (B, C, F, H_new, W_new)
                    clip = resized_clip_reshaped.reshape(batch_size, F_val, C, self.backbone_input_size[0], self.backbone_input_size[1]).permute(0, 2, 1, 3, 4)

                clip = clip.to(next(self.backbone.parameters()).device)
                features = self.backbone(clip) # (batch_size, feature_size)
                temporal_features_for_this_res.append(features)

            # Stack temporal expert features for this resolution:
            # list of (batch_size, feature_size) -> (batch_size, num_temporal_experts, feature_size)
            temporal_features_tensor = torch.stack(temporal_features_for_this_res, dim=1)

            # Temporal Gating for this resolution:
            # Use features from the 'clip_drop_1' expert (index 0 if 1 is first rate) as input.
            temporal_gating_input = temporal_features_tensor[:, self.temporal_sampling_rates.index(1), :]
            raw_temporal_gating_weights = self.temporal_gating_network(temporal_gating_input)
            temporal_gating_weights = torch.softmax(raw_temporal_gating_weights, dim=-1) # (batch_size, num_temporal_experts)
            all_temporal_gating_weights[res_key] = temporal_gating_weights

            # Aggregate temporal features for this resolution using temporal gating weights
            # temporal_features_tensor: (B, E_T, D)
            # temporal_gating_weights: (B, E_T) -> unsqueeze to (B, E_T, 1) for broadcasting
            aggregated_temporal_features = torch.sum(temporal_features_tensor * temporal_gating_weights.unsqueeze(-1), dim=1)
            spatial_expert_aggregated_features.append(aggregated_temporal_features)

        # Stack spatial expert aggregated features:
        # list of (batch_size, feature_size) -> (batch_size, num_spatial_experts, feature_size)
        spatial_expert_features_tensor = torch.stack(spatial_expert_aggregated_features, dim=1)

        # Spatial Gating:
        # Use aggregated features from the first spatial expert (index 0, corresponding to highest resolution like 224x224)
        # as input for spatial gating decision.
        spatial_gating_input = spatial_expert_features_tensor[:, 0, :]
        raw_spatial_gating_weights = self.spatial_gating_network(spatial_gating_input)
        spatial_gating_weights = torch.softmax(raw_spatial_gating_weights, dim=-1) # (batch_size, num_spatial_experts)

        # Final aggregation using spatial gating weights
        # spatial_expert_features_tensor: (B, E_S, D)
        # spatial_gating_weights: (B, E_S) -> unsqueeze to (B, E_S, 1) for broadcasting
        final_aggregated_features = torch.sum(spatial_expert_features_tensor * spatial_gating_weights.unsqueeze(-1), dim=1)

        # Final classification
        logits = self.classifier_head(final_aggregated_features)

        return logits, spatial_gating_weights, all_temporal_gating_weights
