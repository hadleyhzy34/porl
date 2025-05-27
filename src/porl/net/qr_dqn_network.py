import torch
import torch.nn as nn
import torch.nn.functional as F # F may not be needed but good to import
from typing import List

class QRNetwork(nn.Module):
    """
    Quantile Regression Deep Q-Network (QR-DQN) Network.
    Approximates the distribution of returns for each action using a set of quantiles.
    """
    def __init__(self, state_size: int, action_size: int, num_quantiles: int, hidden_sizes: List[int] = [128, 128]):
        """
        Initializes the QRNetwork.

        Args:
            state_size (int): The dimensionality of the input state space.
            action_size (int): The number of possible actions.
            num_quantiles (int): The number of quantiles to predict for the return distribution.
            hidden_sizes (List[int], optional): A list of integers specifying the sizes
                                               of the hidden layers in the feature extractor.
                                               Defaults to [128, 128].
        """
        super().__init__()
        self.action_size = action_size
        self.num_quantiles = num_quantiles

        layers = []
        input_dim = state_size
        for h_dim in hidden_sizes:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.ReLU())
            input_dim = h_dim
        
        # Feature extraction part of the network
        self.feature_extractor = nn.Sequential(*layers)
        
        # Final layer to output quantile values for each action
        # Outputs action_size * num_quantiles values, which will be reshaped
        self.quantile_values_layer = nn.Linear(input_dim, action_size * num_quantiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the QRNetwork.

        Args:
            x (torch.Tensor): Input state tensor with shape (batch_size, state_size).

        Returns:
            torch.Tensor: Predicted quantile values for each action.
                          Shape: (batch_size, action_size, num_quantiles).
        """
        # x shape: (batch_size, state_size)
        # Pass input through the feature extractor
        features = self.feature_extractor(x) # Shape: (batch_size, hidden_sizes[-1])
        
        # Get the raw quantile values from the final layer
        quantile_values = self.quantile_values_layer(features) # Shape: (batch_size, action_size * num_quantiles)
        
        # Reshape to (batch_size, action_size, num_quantiles)
        # The view uses -1 for batch_size to automatically infer it.
        reshaped_quantile_values = quantile_values.view(-1, self.action_size, self.num_quantiles)
        
        return reshaped_quantile_values

    def get_mean_q_values(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the mean Q-values from the predicted quantile distributions.
        This is often used for action selection during evaluation or for epsilon-greedy exploration.

        Args:
            x (torch.Tensor): Input state tensor with shape (batch_size, state_size).

        Returns:
            torch.Tensor: Mean Q-values for each action.
                          Shape: (batch_size, action_size).
        """
        # Get the quantile values from the forward pass
        quantile_values = self.forward(x) # Shape: (batch_size, action_size, num_quantiles)
        
        # Compute the mean across the quantiles dimension (dim=2)
        # This provides an estimate of the expected return for each action.
        mean_q_values = torch.mean(quantile_values, dim=2) # Shape: (batch_size, action_size)
        
        return mean_q_values
