import pdb
from typing import List, Tuple, Union, Callable, Type

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from porl.buffer.replaybuffer import ReplayBuffer
from porl.net.q_network import DuelingQNetwork, QNetwork
from porl.policy.epsilon_greedy_policy import epsilon_greedy_policy
from porl.train.dqn_trainer import DQNTrainer
from porl.utils.logger import Logger


class DDDQNTrainer(DQNTrainer):
    """A trainer for Deep Q-Network (DQN) reinforcement learning.

    Args:
        state_size (int): Size of the state space.
        action_size (int): Number of possible actions.
        device (torch.device): Device to run the networks on.
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

    def learn(self) -> float:
        """one step backward learning procedure

        Returns:
            one step loss value: float
        """
        # Sample directly as tensors
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        with torch.no_grad():
            # pdb.set_trace()
            selected_q_values = self.q_network(next_states)
            next_states_actions = selected_q_values.max(dim=1).indices

            next_q_values = self.target_network(next_states)
            selected_next_q_values = next_q_values.gather(
                1, next_states_actions.unsqueeze(1)
            ).squeeze(1)
            # max_next_q_values = next_q_values.max(dim=1)[0]
            targets = rewards + self.gamma * selected_next_q_values * (1 - dones)

        # with torch.no_grad():
        #     # Double DQN: Use online network to select action, target network to evaluate
        #     next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
        #     next_q_values = (
        #         self.target_network(next_states).gather(1, next_actions).squeeze(1)
        #     )
        #     test_targets = rewards + self.gamma * next_q_values * (1 - dones)

        # pdb.set_trace()

        q_values = self.q_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = F.mse_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(
        self,
        env,
        policy,
        num_episodes: int = 1000,
        max_steps: int = 1000,
        **kwargs,
    ) -> List[float]:
        # collect dataset
        # print(f"collecting dataset\n")
        if "dataset" in kwargs:
            kwargs["dataset"](env, self)

        # pretrain
        # print(f"pretrain dataset\n")
        if "pretrain" in kwargs:
            kwargs["pretrain"](self)
        return super().train(env, policy, num_episodes, max_steps)
