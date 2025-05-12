# src/dqn_project/training/trainer.py
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
from porl.net.q_network import QNetwork
from porl.buffer.replaybuffer import ReplayBuffer
from porl.train.dqn_trainer import DQNTrainer
from porl.utils.logger import Logger
from typing import List, Tuple, Union, Callable, Type


class CQLTrainer(DQNTrainer):
    """A trainer for Conservative Q-Learning (CQL) for offline reinforcement learning.

    Args:
        state_size (int): Size of the state space.
        action_size (int): Number of possible actions.
        device (torch.device): Device to run the networks on.
        dataset (List[Tuple]): Offline dataset of experiences (s, a, r, s', done).
        log_dir (str): Directory to store logs.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        gamma: float,
        epsilon: float,
        epsilon_min: float,
        epsilon_decay: float,
        update_target_freq: int,
        device: torch.device,
        # network: nn.Module,
        network: Type[nn.Module],
        log_dir: str = "logs",
        num_epochs: int = 1000,
        threshold: float = 0.1,
        alpha: int = 1,
    ):
        super().__init__(
            state_size,
            action_size,
            gamma,
            epsilon,
            epsilon_min,
            epsilon_decay,
            update_target_freq,
            device,
            network,
            log_dir,
        )
        self.num_epochs = num_epochs
        self.threshold = threshold
        self.alpha = alpha

    def compute_cql_penalty(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Compute the conservative penalty term.

        Args:
            states (torch.Tensor): Batch of states.
            actions (torch.Tensor): Batch of actions from the dataset.

        Returns:
            torch.Tensor: Conservative penalty.
        """
        batch_size = states.size(0)
        # Compute Q-values for all actions
        q_values = self.q_network(states)  # Shape: (batch_size, action_size)

        # Approximate E_{a ~ mu(a|s)} [Q(s, a)] using log-sum-exp over all actions (uniform mu)
        log_sum_exp = torch.logsumexp(q_values, dim=1) - torch.log(
            torch.tensor(self.action_size, dtype=torch.float32, device=self.device)
        )

        # Compute Q(s, a) for the actions in the dataset
        q_dataset = q_values[range(batch_size), actions]

        # Conservative penalty: E_{a ~ mu} [Q(s, a)] - E_{dataset} [Q(s, a)]
        penalty = log_sum_exp - q_dataset
        return penalty.mean()

    def learn(self):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )
        
        # Compute TD loss
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            next_actions = next_q_values.argmax(dim=1, keepdim=True)
            target_q_values = next_q_values.gather(1, next_actions).squeeze(1)
            targets = rewards + self.gamma * target_q_values * (1 - dones)

        q_values = self.q_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        td_loss = F.mse_loss(q_values, targets)

        # Compute conservative penalty
        cql_penalty = self.compute_cql_penalty(states, actions)

        # Total loss
        loss = td_loss + self.alpha * cql_penalty

        # Update Q-network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Log both losses if you have a logger
        if hasattr(self, "logger"):
            self.logger.writer.add_scalar("Loss/TD", td_loss.item(), self.training_step)
            self.logger.writer.add_scalar("Loss/CQLPenalty", cql_penalty.item(), self.training_step)
            self.logger.writer.add_scalar("Loss/Total", loss.item(), self.training_step)
            self.training_step += 1

        return loss.item()

    def get_action(self, state: np.ndarray) -> int:
        """Select an action for evaluation.

        Args:
            state (np.ndarray): Current state.

        Returns:
            int: Selected action.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            action = q_values.argmax(dim=1).item()
        return action
