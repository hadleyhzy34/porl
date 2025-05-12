import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from typing import List


class BehaviorPolicy(nn.Module):
    """A model to approximate the behavior policy pi_b(a|s) as a categorical distribution.

    Args:
        state_size (int): Size of the input state vector.
        action_size (int): Number of possible actions.
        hidden_sizes (list[int], optional): Sizes of hidden layers.
    """

    def __init__(
        self, state_size: int, action_size: int, hidden_sizes: List[int] = [64, 128]
    ):
        super(BehaviorPolicy, self).__init__()
        self.action_size = action_size
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], action_size),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """forward from state to action distribution

        Args:
            state: torch.tensor, (b,s)

        Returns:
            torch.tensor, (b,a), normalized Probabilities over actions
        """
        logits = self.network(state)  # Shape: (batch_size, action_size)
        return F.softmax(logits, dim=-1)  # Probabilities over actions

    def sample(self, state: torch.Tensor, threshold: float = 0.1) -> torch.Tensor:
        """Sample actions with probability above the threshold.

        Args:
            state (torch.Tensor): Input states.
            threshold (float): Probability threshold for action selection.

        Returns:
            torch.Tensor: Mask of allowed actions.
        """
        # pdb.set_trace()
        probs = self(state)  # Shape: (batch_size, action_size)
        mask = (probs > threshold).float()  # Shape: (batch_size, action_size)
        return mask
