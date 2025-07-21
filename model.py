from typing import Dict, List

import torch
import torch.nn as nn
from torchvision.models.video import r3d_18

class MoE_Action_Model(nn.Module):
    """
    A Mixture-of-Experts (MoE) model for action recognition.

    This model uses multiple "expert" pathways, each corresponding to a different
    temporal sampling rate of the input video. A lightweight gating network
    learns to assign weights to these experts, and their outputs are combined
    to make a final prediction.
    """
    def __init__(self, num_classes: int, temporal_sampling_rates: List[int],
                 pretrained: bool = True):
        """
        Args:
            num_classes (int): The number of action classes for the final prediction.
            temporal_sampling_rates (List[int]): The list of clip_step values, one for each expert.
            pretrained (bool): If True, loads weights pre-trained on the Kinetics dataset.
        """
        super().__init__()
        self.temporal_sampling_rates = temporal_sampling_rates
        self.num_experts = len(temporal_sampling_rates)

        # 1. --- The Shared-Weight 3D CNN Backbone ---
        # We use a single ResNet-3D model as our backbone. The features from each
        # temporal sampling expert will be passed through this same backbone.
        self.backbone = r3d_18(weights='KINETICS400_V1' if pretrained else None)

        # We'll use the backbone as a feature extractor. The output of the `avgpool`
        # layer is a 512-dimensional feature vector.
        backbone_out_features = self.backbone.fc.in_features
        # We replace the final fully connected layer with an identity function,
        # so the output of the backbone is the feature vector itself.
        self.backbone.fc = nn.Identity()

        # 2. --- The Lightweight Gating Network ---
        # This network decides how much to trust each expert for a given video.
        # It takes the feature vector from the first expert as its input.
        self.gating_network = nn.Sequential(
            nn.Linear(backbone_out_features, self.num_experts),
            nn.Softmax(dim=1)
        )

        # 3. --- The Classification Head ---
        # This final layer takes the mixed feature vector from the experts and
        # maps it to the number of action classes.
        self.classification_head = nn.Linear(backbone_out_features, num_classes)


    def forward(self, x: Dict[str, torch.Tensor]) -> (torch.Tensor, torch.Tensor):
        """
        Defines the forward pass of the model.

        Args:
            x (Dict[str, torch.Tensor]): A dictionary of video clips from our VideoDataset.
                                         The keys are e.g., 'clip_rate_8', 'clip_rate_16'.

        Returns:
            A tuple containing:
            - logits (torch.Tensor): The final class predictions.
            - routing_weights (torch.Tensor): The weights assigned by the gating network.
        """
        # --- Feature Extraction for Each Expert ---
        expert_features = []
        # We process the clips in the order defined by the sampling rates
        for rate in self.temporal_sampling_rates:
            clip = x[f'clip_rate_{rate}']
            features = self.backbone(clip)
            expert_features.append(features)

        # --- Gating Mechanism ---
        # The input to the gating network is the feature from the first expert
        # (assuming the first expert has the most fine-grained temporal information).
        gating_input = expert_features[0].detach() # Detach to prevent gradients from flowing back
        routing_weights = self.gating_network(gating_input) # Shape: (batch_size, num_experts)

        # --- Weighted Combination of Expert Outputs ---
        # We combine the expert features using the learned routing weights.
        # Reshape weights for broadcasting: (batch_size, num_experts) -> (batch_size, num_experts, 1)
        routing_weights_reshaped = routing_weights.unsqueeze(-1)

        # Stack expert features: list of (batch_size, features) -> (batch_size, num_experts, features)
        stacked_expert_features = torch.stack(expert_features, dim=1)

        # Perform the weighted sum
        mixed_features = (routing_weights_reshaped * stacked_expert_features).sum(dim=1)

        # --- Final Classification ---
        logits = self.classification_head(mixed_features)

        return logits, routing_weights
