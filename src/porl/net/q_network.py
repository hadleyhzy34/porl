import torch
import torch.nn as nn
import abc
from typing import Any, Tuple, Dict, Union, List, Optional  # For type hinting
import numpy as np


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_sizes=[64, 128, 64]):
        super(QNetwork, self).__init__()
        layers = []
        cur_size = state_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(cur_size, hidden_size))
            layers.append(nn.ReLU())
            cur_size = hidden_size

        layers.append(nn.Linear(cur_size, action_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """forward call
        Args:
            x (torch.Tensor): current state, (b,s)

        Returns:
            q_values(torch.Tensor): estimated q values for each action, (b,a)
        """
        return self.model(x)


class DuelingQNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_sizes=[64, 128, 64]):
        super(DuelingQNetwork, self).__init__()
        layers = []
        cur_size = state_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(cur_size, hidden_size))
            layers.append(nn.ReLU())
            cur_size = hidden_size

        # Value stream
        self.value = nn.Sequential(
            nn.Linear(64, 1),  # Scalar V(s)
        )

        # advantage stream
        self.advantage = nn.Sequential(nn.Linear(64, action_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Dueling Network: value + advantage

        Args:
            state(torch.Tensor): current state, (b,s)
        Return:
            q_values(torch.Tensor): estimated q values for each action, (b,a)
        """
        features = self.model(x)
        value = self.value(features)
        advantage = self.advantage(features)

        # normalization trick
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values
