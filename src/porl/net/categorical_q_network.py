import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class CategoricalQNetwork(nn.Module):
    """A categorical Q-network for approximating the distribution of returns.
    Args:
        state_size (int): Size of the input state vector.
        action_size (int): Number of possible actions.
        num_atoms (int): Number of support points for the distribution.
        v_min (float): Minimum value of the support.
        v_max (float): Maximum value of the support.
        hidden_sizes (list[int], optional): Sizes of hidden layers.
    """

    def __init__(
        self,
        state_size,
        action_size,
        atom_size=51,
        v_min=-10,
        v_max=10,
        hidden_sizes=[128, 128],
    ):
        super().__init__()
        self.action_size = action_size
        self.atom_size = atom_size
        self.v_min = v_min
        self.v_max = v_max
        self.support = torch.linspace(v_min, v_max, atom_size)
        layers = []
        input_size = state_size
        for h in hidden_sizes:
            layers.append(nn.Linear(input_size, h))
            layers.append(nn.ReLU())
            input_size = h
        self.feature = nn.Sequential(*layers)
        self.fc = nn.Linear(input_size, action_size * atom_size)

    def forward(self, x):
        # pdb.set_trace()
        batch_size = x.size(0)
        x = self.feature(x)
        x = self.fc(x)
        x = x.view(batch_size, self.action_size, self.atom_size)
        x = F.log_softmax(x, dim=2)  # Log-probabilities for numerical stability
        return x

    def get_support(self, device):
        return self.support.to(device)

    def get_q_values(self, x: torch.Tensor) -> torch.Tensor:
        """Compute expected Q values from the distribution

        Args:
            x: state, torch.Tensor, (b,s)

        Returns:
            q_values: torch.Tensor, (b,a)
        """
        """Compute expected Q-values from the distribution."""
        # pdb.set_trace()
        probs = self(x)  # Shape: (batch_size, action_size, num_atoms)
        # Expand support to match batch size and action size
        support = self.support.expand(
            probs.size(0), self.action_size, self.atom_size
        ).to(x.device)
        q_values = (probs * support).sum(dim=-1)  # Shape: (batch_size, action_size)
        return q_values
