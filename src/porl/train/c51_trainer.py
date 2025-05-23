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
        atom_size: int = 51,
        v_min: float = -10,
        v_max: float = 10,
        network_hidden_sizes: List[int] = [128, 128],  # Added parameter
        log_dir: str = "logs",
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
            # Pass hidden_sizes to CategoricalQNetwork
            network=lambda s, a: CategoricalQNetwork(
                s, a, atom_size, v_min, v_max, hidden_sizes=network_hidden_sizes
            ),
            log_dir=log_dir,
        )
        self.atom_size = atom_size
        if self.atom_size < 2:
            raise ValueError("atom_size must be at least 2.")
        self.v_min = v_min
        self.v_max = v_max
        # Ensure atom_size is valid before calculating delta_z
        self.delta_z = (v_max - v_min) / (atom_size - 1)
        self.support = torch.linspace(v_min, v_max, atom_size).to(device)

    def learn(self):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )
        # pdb.set_trace()
        # Removed pdb.set_trace()
        batch_size = states.size(0)

        with torch.no_grad():
            # Calculate target distribution for the next state (using the target network)
            # 1. Get log-probabilities from target network, then convert to probabilities
            # self.target_network(next_states) returns log-probs: [Batch, Action_Size, Atom_Size]
            # .exp() converts log-probs to probabilities: [Batch, Action_Size, Atom_Size]
            next_log_probs = self.target_network(next_states)
            next_probs = next_log_probs.exp()

            # 2. Calculate Q-values for next states by taking the expectation over the support
            # self.support is [Atom_Size]. Unsqueeze to allow broadcasting with next_probs.
            # next_q_values shape: [Batch, Action_Size]
            next_q_values = torch.sum(
                next_probs * self.support.unsqueeze(0).unsqueeze(0), dim=2
            )

            # 3. Select optimal next actions based on these Q-values (greedy policy for target)
            # next_optimal_actions shape: [Batch]
            next_optimal_actions = next_q_values.argmax(1)

            # 4. Get the probability distribution for these optimal next actions
            # next_dist_optimal shape: [Batch, Atom_Size]
            # given each next state optimal actions, their q value distributions
            next_dist_optimal = next_probs[range(batch_size), next_optimal_actions]

            # Perform the Bellman update for each atom in the support
            # t_z_j = r + gamma * z_j for non-terminal states
            # t_z_j = r for terminal states
            # self.support is [Atom_Size]. rewards is [Batch]. dones is [Batch].
            # After unsqueezing: rewards -> [B,1], self.support -> [1,Atom_Size], dones -> [B,1]
            # t_z (projected_support) shape: [Batch, Atom_Size]
            projected_support = rewards.unsqueeze(
                1
            ) + self.gamma * self.support.unsqueeze(0) * (
                1
                - dones.unsqueeze(1).float()  # Ensure dones is float for multiplication
            )

            # Clamp the projected support values to be within [v_min, v_max]
            projected_support_clamped = projected_support.clamp(self.v_min, self.v_max)

            # Calculate projection indices (b, l, u) for distributing probabilities
            # These indices determine how the probability mass of each atom in next_dist_optimal
            # is distributed to the atoms of the target distribution (proj_dist).
            # b_j = (t_z_j - v_min) / delta_z
            # b shape: [Batch, Atom_Size]
            b = (projected_support_clamped - self.v_min) / self.delta_z
            l = b.floor().long()  # Lower bound atom index
            u = b.ceil().long()  # Upper bound atom index

            # Initialize the target projection distribution (m in the C51 paper)
            # proj_dist shape: [Batch, Atom_Size]
            proj_dist = torch.zeros_like(next_dist_optimal, device=states.device)

            # Ensure l and u are within bounds [0, atom_size - 1] for indexing scatter_add_
            l_clamped = l.clamp(0, self.atom_size - 1)
            u_clamped = u.clamp(0, self.atom_size - 1)

            # Distribute probabilities to the target distribution (proj_dist)
            # This performs the projection of the next state's distribution (next_dist_optimal)
            # onto the support defined by projected_support_clamped.
            # It handles two cases for each atom j in next_dist_optimal:
            # 1. Non-exact hits (l_j != u_j): Probability mass p_j(s_t+1, a*) is distributed proportionally
            #    to atoms l_j and u_j based on their proximity to b_j.
            #    Mass to l_j: p_j(s_t+1, a*) * (u_j - b_j)
            #    Mass to u_j: p_j(s_t+1, a*) * (b_j - l_j)
            # 2. Exact hits (l_j == u_j): Probability mass p_j(s_t+1, a*) is assigned entirely to atom l_j.

            # Mask for non-exact hits (where the projected atom b_j is not an integer)
            non_exact_hit_mask = l != u  # Shape: [Batch, Atom_Size]
            # Mask for exact hits (where b_j is an integer, so l_j == u_j)
            exact_hit_mask = ~non_exact_hit_mask  # Shape: [Batch, Atom_Size]

            # Calculate contributions for non-exact hits
            # `next_dist_optimal` contains the probabilities p_j(s_t+1, a*)
            val_l_non_exact = next_dist_optimal * (u.float() - b)
            val_u_non_exact = next_dist_optimal * (b - l.float())

            # Add contributions for non-exact hits to proj_dist, applying the mask
            proj_dist.scatter_add_(
                dim=1, index=l_clamped, src=val_l_non_exact * non_exact_hit_mask
            )
            proj_dist.scatter_add_(
                dim=1, index=u_clamped, src=val_u_non_exact * non_exact_hit_mask
            )

            # Add contributions for exact hits to proj_dist, applying the mask
            # For exact hits, the entire probability mass next_dist_optimal[j] goes to atom l_clamped[j]
            proj_dist.scatter_add_(
                dim=1, index=l_clamped, src=next_dist_optimal * exact_hit_mask
            )

        # Get current Q-network's predicted log-probability distribution for (states, actions)
        # `current_log_probs` has shape [Batch, Action_Size, Atom_Size] (log probabilities)
        current_log_probs = self.q_network(states)
        # Select the log-probabilities for the actions taken: log p(s_t, a_t)
        # actions is [Batch] or [B,1]. If [B,1], squeeze. If [B], ensure long.
        # log_p_taken_actions shape: [Batch, Atom_Size]
        log_p_taken_actions = current_log_probs[
            range(batch_size), actions.squeeze().long()
        ]

        # Calculate the cross-entropy loss between the target distribution (proj_dist)
        # and the Q-network's predicted distribution for the taken actions (log_p_taken_actions).
        # Loss = - sum_i (target_prob_i * log(predicted_prob_i))
        # proj_dist contains target probabilities.
        # log_p_taken_actions contains log of predicted probabilities.
        # To prevent log(0) if any predicted probability is zero after exp(), clamp is used.
        loss = (
            -(proj_dist * log_p_taken_actions.exp().clamp(min=1e-8).log()).sum(1).mean()
        )

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
                # Removed pdb.set_trace()
                # Removed commented-out print(action)
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

            # Removed pdb.set_trace()
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
