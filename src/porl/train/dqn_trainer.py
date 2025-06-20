import ipdb
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
    """trainer framework for dqn learning

    Attributes:
        state_size: state observation dimension
        action_size: action observation space dimension
        device: cpu or gpu
        gamma: gamma parameter
        epsilon: current random policy probability
        epsilon_min: minimum random policy probability
        epsilon_decay: random policy prob decay rate
        update_target_freq: udpate rate
        q_network: q network
        target_network: q network as target
        optimizer: learning optimizer for q network
        logger: directory for logs
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        gamma: float,
        epsilon: float,
        epsilon_min: float,
        epsilon_decay: float,
        learning_rate: float,
        update_target_freq: int,
        device: torch.device,
        # network: nn.Module,
        network: Type[nn.Module] | None,
        log_dir: str = "logs",
        replay_buffer=None,
        transition_learning_step=10000,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device

        # Hyperparameters settings
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.update_target_freq = update_target_freq
        self.training_learning_step = transition_learning_step

        # Initialize networks and optimizer
        if network is not None:
            self.q_network = network(state_size, action_size).to(device)
            self.target_network = network(state_size, action_size).to(device)
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.target_network.eval()
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.0005)

        # Initialize replay buffer
        # pdb.set_trace()
        if replay_buffer is None:
            self.replay_buffer = ReplayBuffer(100000, (self.state_size,), device)

        self.batch_size = 64

        # Initialize logger
        self.logger = Logger(log_dir=log_dir)

        # Log hyperparameters
        hparams = {
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "batch_size": self.batch_size,
            "epsilon_decay": self.epsilon_decay,
            "update_target_freq": self.update_target_freq,
        }
        self.logger.log_hyperparameters(hparams)

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
            next_q_values = self.target_network(next_states)
            max_next_q_values = next_q_values.max(dim=1)[0]
            targets = rewards + self.gamma * max_next_q_values * (1 - dones)

        q_values = self.q_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = F.mse_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train_online(
        self, env, policy=None, num_episodes: int = 1000, max_steps: int = 1000
    ) -> List[float]:
        rewards_history = []

        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0

            for step in range(max_steps):
                # action = epsilon_greedy_policy(
                #     state,
                #     self.epsilon,
                #     self.q_network,
                #     self.action_size,
                #     self.device,
                # )
                action = self.select_action(state)
                # pdb.set_trace()
                # print(action)
                next_state, reward, done, truncated, _ = env.step(action)
                done = done or truncated

                self.replay_buffer.push(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward

                # ipdb.set_trace()
                # Log the reward for this step (loss will be logged after training)
                self.logger.log_step(episode, step, reward, None, self.epsilon)
                # print(f"episode:{episode},step:{step},reward:{reward}")

                # if len(self.replay_buffer) >= self.batch_size:
                if len(self.replay_buffer) >= self.training_learning_step:
                    if policy is None:
                        loss = self.learn()
                    else:
                        loss = policy()
                    if self.logger is not None:
                        self.logger.log_step(episode, step, reward, loss, self.epsilon)

                if done:
                    break

            # pdb.set_trace()
            # ipdb.set_trace()
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

    def train_offline(self, policy=None, num_iterations: int = 10000) -> List[float]:
        losses = []

        for self.training_step in range(num_iterations):
            if policy is None:
                loss = self.learn()
            else:
                loss = policy()

            # Log the reward for this step (loss will be logged after training)
            self.logger.log_loss(self.training_step, loss)

            # Update target network
            if self.training_step % self.update_target_freq == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())

            losses.append(loss)

            if self.training_step % 10 == 0:
                print(f"step {self.training_step}, Loss: {loss}")

        self.logger.close()
        return losses

    def select_action(self, state: np.ndarray) -> int:
        """
        Selects an action using an epsilon-greedy policy based on mean Q-values.
        This method overrides the `select_action` method in DQNTrainer to use
        the mean of the quantile distribution for action selection, which is
        appropriate for QR-DQN.

        Args:
            state (np.ndarray): The current state of the environment.
                                Shape: (state_size,)

        Returns:
            int: The selected action (an integer from 0 to action_size-1).
        """
        # ipdb.set_trace()

        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)

        state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.q_network(state_tensor)

        # Removed commented-out print statement
        return q_values.argmax().item()
