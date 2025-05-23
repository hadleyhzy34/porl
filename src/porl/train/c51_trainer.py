import torch
import torch.nn.functional as F
from typing import List, Tuple, Union, Callable, Type
import numpy as np
from porl.train.dqn_trainer import DQNTrainer
from porl.net.categorical_q_network import CategoricalQNetwork
from porl.policy.epsilon_greedy_policy import epsilon_greedy_categorical_policy
import pdb


class C51Trainer(DQNTrainer):
    def __init__(
        self,
        state_size,
        action_size,
        gamma,
        epsilon,
        epsilon_min,
        epsilon_decay,
        update_target_freq,
        device,
        atom_size=51,
        v_min=-10,
        v_max=10,
        log_dir="logs",
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
            network=lambda s, a: CategoricalQNetwork(s, a, atom_size, v_min, v_max),
            log_dir=log_dir,
        )
        self.atom_size = atom_size
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (atom_size - 1)
        self.support = torch.linspace(v_min, v_max, atom_size).to(device)

    def learn(self):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )
        pdb.set_trace()
        batch_size = states.size(0)
        with torch.no_grad():
            next_dist = self.target_network(next_states).exp()  # [B, A, N]
            next_q = torch.sum(next_dist * self.support, dim=2)
            next_action = next_q.argmax(1)
            next_dist = next_dist[range(batch_size), next_action]

            t_z = rewards.unsqueeze(1) + self.gamma * self.support.unsqueeze(0) * (
                1 - dones.unsqueeze(1)
            )
            t_z = t_z.clamp(self.v_min, self.v_max)
            b = (t_z - self.v_min) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()

            proj_dist = torch.zeros(next_dist.size(), device=states.device)
            for i in range(self.atom_size):
                idx_l = l == i
                idx_u = u == i
                proj_dist[idx_l] += next_dist[idx_l] * (u.float() - b)[idx_l]
                proj_dist[idx_u] += next_dist[idx_u] * (b - l.float())[idx_u]

        dist = self.q_network(states)
        log_p = dist[range(batch_size), actions.squeeze().long()]
        loss = -(proj_dist * log_p.exp().clamp(min=1e-8).log()).sum(1).mean()

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
                action = epsilon_greedy_categorical_policy(
                    state,
                    self.epsilon,
                    self.q_network,
                    self.action_size,
                    self.device,
                )
                # pdb.set_trace()
                # print(action)
                next_state, reward, done, truncated, _ = env.step(action)
                done = done or truncated

                self.replay_buffer.push(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward

                # Log the reward for this step (loss will be logged after training)
                self.logger.log_step(episode, step, reward, None, self.epsilon)

                if len(self.replay_buffer) >= self.batch_size:
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
