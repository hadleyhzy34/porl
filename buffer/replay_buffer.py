import random
from collections import deque

import numpy as np
import torch


class ReplayBuffer:
    """A replay buffer that stores experiences in contiguous NumPy arrays for efficient sampling.

    Args:
        capacity (int): Maximum number of experiences to store.
        state_shape (tuple): Shape of the state (e.g., (8,) for LunarLander-v2).
        device (torch.device): Device to move sampled tensors to.
    """

    def __init__(
        self, capacity: int, state_shape: tuple[int, ...], device: torch.device
    ):
        self.capacity = capacity
        self.state_shape = state_shape
        self.device = device
        self.size = 0
        self.position = 0

        # Pre-allocate NumPy arrays for each component
        self.states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add an experience to the buffer."""
        # Store the experience at the current position
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = float(done)  # Convert bool to float for consistency

        # Update position and size
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a batch of experiences and return them as PyTorch tensors.

        Args:
            batch_size (int): Number of experiences to sample.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as PyTorch tensors.
        """
        indices = np.random.choice(self.size, batch_size, replace=False)

        # Convert NumPy arrays directly to PyTorch tensors
        states = torch.from_numpy(self.states[indices]).float().to(self.device)
        actions = torch.from_numpy(self.actions[indices]).long().to(self.device)
        rewards = torch.from_numpy(self.rewards[indices]).float().to(self.device)
        next_states = (
            torch.from_numpy(self.next_states[indices]).float().to(self.device)
        )
        dones = torch.from_numpy(self.dones[indices]).float().to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """Return the current number of experiences in the buffer."""
        return self.size


# class replaybuffer:
#     def __init__(self, capacity):
#         self.buffer = deque(maxlen=capacity)
#
#     def push(self, state, action, reward, next_state, done):
#         self.buffer.append((state, action, reward, next_state, done))
#
#     def sample(self, batch_size):
#         return random.sample(self.buffer, batch_size)
#
#     def __len__(self):
#         return len(self.buffer)
