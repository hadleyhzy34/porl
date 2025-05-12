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


class BCQTrainer(DQNTrainer):
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
        behavior_policy: Type[nn.Module],
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

        # behavior policy
        self.behavior_policy = behavior_policy(state_size, action_size).to(device)
        self.behavior_optimizer = optim.Adam(
            self.behavior_policy.parameters(), lr=0.0005
        )

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
