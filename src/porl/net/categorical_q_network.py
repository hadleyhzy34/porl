import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from typing import List


class CategoricalQNetwork(nn.Module):
    """A categorical Q-network for approximating the distribution of returns.
    Args:
        state_size (int): Size of the input state vector.
        action_size (int): Number of possible actions.
        atom_size (int): Number of support points (atoms) for the discrete distribution.
                         Defines the resolution of the value distribution.
        v_min (float): Minimum value of the support range. The smallest possible return value
                       the distribution can represent.
        v_max (float): Maximum value of the support range. The largest possible return value
                       the distribution can represent.
        hidden_sizes (list[int], optional): Sizes of hidden layers in the MLP.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        atom_size: int = 51,  # Number of atoms in the categorical distribution
        v_min: float = -10,  # Minimum value of the support
        v_max: float = 10,  # Maximum value of the support
        hidden_sizes: List[int] = [128, 128],
    ):
        super().__init__()
        self.action_size = action_size
        self.atom_size = atom_size
        self.v_min = v_min
        self.v_max = v_max
        # self.support: A tensor representing the discrete values (atoms) z_i that the
        # distribution is defined over. These are fixed, evenly spaced points
        # between v_min and v_max. Shape: [atom_size]
        self.support = torch.linspace(v_min, v_max, atom_size)
        layers = []
        input_size = state_size
        for h in hidden_sizes:
            layers.append(nn.Linear(input_size, h))
            layers.append(nn.ReLU())
            input_size = h
        self.feature = nn.Sequential(*layers)
        self.fc = nn.Linear(input_size, action_size * atom_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the network.
        The network processes the input state and outputs a discrete probability
        distribution over the support atoms for each possible action.

        Args:
            x (torch.Tensor): Input state tensor, shape (batch_size, state_size).

        Returns:
            torch.Tensor: Log-probabilities of the value distribution for each action.
                          Shape: (batch_size, action_size, atom_size).
                          `log_softmax` is applied along the atom dimension (dim=2).
                          Using log-probabilities improves numerical stability during
                          the subsequent cross-entropy loss calculation in the C51 algorithm.
        """
        # Removed pdb.set_trace()
        batch_size = x.size(0)
        # Pass state through feature layers (MLP)
        x = self.feature(x)
        # Final linear layer to get scores for each atom of each action
        x = self.fc(x)
        # Reshape to (batch_size, action_size, atom_size) to separate distributions per action
        x = x.view(batch_size, self.action_size, self.atom_size)
        # Apply log_softmax along the atom dimension (dim=2) to get log-probabilities.
        # This ensures that for each action, the probabilities of its atoms sum to 1 (in probability space),
        # and using log-probabilities is numerically more stable for the cross-entropy loss.
        x = F.log_softmax(x, dim=2)
        return x

    def get_support(self, device: torch.device) -> torch.Tensor:
        return self.support.to(device)

    def get_q_values(self, x: torch.Tensor) -> torch.Tensor:
        """Compute expected Q values from the distribution

        Args:
            x: state, torch.Tensor, (b,s)

        Returns:
            q_values: torch.Tensor, (b,a)
        """
        # Removed pdb.set_trace()
        # pdb.set_trace()
        # Call the forward pass to get log-probabilities for each action's value distribution.
        # log_probs has shape: (batch_size, action_size, atom_size)
        log_probs = self(x)

        # Convert log-probabilities to actual probabilities using exp().
        # probs has shape: (batch_size, action_size, atom_size)
        probs = log_probs.exp()

        # Expand self.support to match the shape of probs for element-wise multiplication.
        # self.support is originally [atom_size].
        # support_expanded will have shape: [batch_size, action_size, atom_size]
        # and will be on the same device as input x.
        support_expanded = self.support.expand_as(probs).to(x.device)

        # Calculate the expected Q-value for each action.
        # This is done by taking the dot product of the probability distribution (probs)
        # and the support values (support_expanded) for each action.
        # Q(s,a) = sum_i (p_i(s,a) * z_i), where z_i are the atom values in support.
        # q_values has shape: (batch_size, action_size)
        q_values = (probs * support_expanded).sum(dim=-1)
        return q_values
