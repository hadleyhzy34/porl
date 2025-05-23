import pdb
from typing import List, Tuple, Union, Callable, Type

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from porl.buffer.replaybuffer import PrioritizedReplayBuffer
from porl.buffer.prioritized_replay_buffer import PrioritizedReplayBuffer
from porl.net.q_network import DuelingQNetwork, QNetwork
from porl.policy.epsilon_greedy_policy import epsilon_greedy_policy
from porl.train.dqn_trainer import DQNTrainer
from porl.utils.logger import Logger


class PERTrainer(DQNTrainer):
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
            replay_buffer="PER",
        )
        self.num_epochs = num_epochs
        self.threshold = threshold

        # In your agent or trainer __init__
        # self.memory = ReplayBuffer(capacity) # OLD
        self.memory = PrioritizedReplayBuffer(
            100000, alpha=0.6, beta_start=0.4, beta_frames=100000
        )  # NEW
        self.max_initial_priority = 1.0  # For adding new experiences

    def learn(self) -> float:
        """one step backward learning procedure

        Returns:
            one step loss value: float
        """
        # pdb.set_trace()
        # Sample directly as tensors,sample and get IS weights and indices
        states, actions, rewards, next_states, dones, is_weights, tree_indices = (
            self.memory.sample(self.batch_size)
        )
        # states, actions, rewards, next_states, dones = self.replay_buffer.sample(
        #     self.batch_size
        # )
        #
        # Convert to tensors (PyTorch example)
        states = torch.FloatTensor(states).to(self.device)
        actions = (
            torch.LongTensor(actions).unsqueeze(1).to(self.device)
        )  # Ensure actions are [batch_size, 1]
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        is_weights = torch.FloatTensor(is_weights).unsqueeze(1).to(self.device)

        with torch.no_grad():
            # pdb.set_trace()
            selected_q_values = self.q_network(next_states)
            next_states_actions = selected_q_values.max(dim=1).indices

            next_q_values = self.target_network(next_states)
            selected_next_q_values = next_q_values.gather(
                1, next_states_actions.unsqueeze(1)
            )  # (b,1)
            # max_next_q_values = next_q_values.max(dim=1)[0]
            targets = (
                rewards + self.gamma * selected_next_q_values * (1 - dones)
            ).squeeze(1)  # (b,)

        q_values = self.q_network(states)
        q_values = q_values.gather(1, actions).squeeze(1)

        # weighted MSE loss
        # Loss_i = IS_weight_i * (TD_error_i)^2
        loss = (is_weights * (q_values - targets) ** 2).mean()
        # loss = F.mse_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # after TD-learning, update sampled exp, Get the absolute TD errors for updating priorities
        # We need these as raw numbers, not tensors on GPU, for the SumTree
        abs_td_errors = (
            (q_values - targets).abs().detach().cpu().numpy().squeeze()
        )  # Squeeze if it's [batch_size, 1]
        self.memory.update_priorities(tree_indices, abs_td_errors)

        return loss.item()

    def train_online(
        self, env, policy=None, num_episodes: int = 1000, max_steps: int = 1000
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

                self.memory.add(
                    self.max_initial_priority, state, action, reward, next_state, done
                )
                state = next_state
                episode_reward += reward

                # Log the reward for this step (loss will be logged after training)
                self.logger.log_step(episode, step, reward, None, self.epsilon)

                if len(self.memory) >= self.batch_size:
                    if policy is None:
                        loss = self.learn()
                    else:
                        loss = policy()
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
