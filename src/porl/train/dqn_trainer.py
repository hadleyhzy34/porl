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
from porl.utils.logger import Logger


class DQNTrainer:
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
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device

        # Hyperparameters settings
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.update_target_freq = update_target_freq

        # Initialize networks and optimizer
        self.q_network = network(state_size, action_size).to(device)
        self.target_network = network(state_size, action_size).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.0005)

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(100000, (self.state_size,), device)
        self.batch_size = 64

        # Initialize logger
        self.logger = Logger(log_dir=log_dir)

        # Log hyperparameters
        hparams = {
            "learning_rate": 0.0005,
            "gamma": self.gamma,
            "batch_size": self.batch_size,
            "epsilon_decay": self.epsilon_decay,
            "update_target_freq": self.update_target_freq,
        }
        self.logger.log_hyperparameters(hparams)

    def train(
        self, env, policy, num_episodes: int = 1000, max_steps: int = 1000
    ) -> List[float]:
        rewards_history = []

        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0

            for step in range(max_steps):
                action = epsilon_greedy_policy(
                    state,
                    self.epsilon,
                    self.q_network,
                    self.action_size,
                    self.device,
                )
                next_state, reward, done, truncated, _ = env.step(action)
                done = done or truncated

                self.replay_buffer.push(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward

                # Log the reward for this step (loss will be logged after training)
                self.logger.log_step(episode, step, reward, None, self.epsilon)

                if len(self.replay_buffer) >= self.batch_size:
                    loss = policy(self)
                    if self.logger is not None:
                        self.logger.log_step(episode, step, reward, loss, self.epsilon)

                if done:
                    break

            # pdb.set_trace()
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            if episode % self.update_target_freq == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())

            # Log episode summary
            self.logger.log_episode(episode)
            rewards_history.append(episode_reward)

            if episode % 10 == 0:
                print(
                    f"Episode {episode}, Reward: {episode_reward:.2f}, Epsilon: {self.epsilon:.3f}"
                )

        env.close()
        self.logger.close()
        return rewards_history
