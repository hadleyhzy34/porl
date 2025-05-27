import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from typing import List


class QRDQNNetwork(nn.Module):
    def __init__(self, state_size, action_size, num_quantiles=51, hidden_size=512):
        super(QRDQNNetwork, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.num_quantiles = num_quantiles

        # Feature extraction layers
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # Quantile value layer
        self.value_layer = nn.Linear(hidden_size, action_size * num_quantiles)

        # Quantile fractions (τ)
        self.register_buffer(
            "quantile_fractions",
            torch.arange(0.5 / num_quantiles, 1, 1 / num_quantiles),
        )

    def forward(self, state):
        batch_size = state.size(0)

        # Extract features
        features = self.feature_layer(state)

        # Get quantile values
        quantile_values = self.value_layer(features)

        # Reshape to [batch_size, action_size, num_quantiles]
        quantile_values = quantile_values.view(
            batch_size, self.action_size, self.num_quantiles
        )

        return quantile_values

    def get_q_values(self, state):
        """Get expected Q-values by averaging over quantiles"""
        quantile_values = self.forward(state)
        return quantile_values.mean(dim=2)
