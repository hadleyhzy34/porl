import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
import random
from collections import deque
import matplotlib.pyplot as plt
import ipdb
from porl.net.iqn_network import IQNNetwork


class IQNAgent:
    def __init__(
        self,
        state_size,
        action_size,
        lr=5e-4,
        gamma=0.99,
        buffer_size=100000,
        batch_size=32,
        target_update=1000,
        n_quantiles=8,
        kappa=1.0,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.n_quantiles = n_quantiles
        self.kappa = kappa  # Huber loss threshold

        # Networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = IQNNetwork(state_size, action_size).to(self.device)
        self.target_network = IQNNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Replay buffer
        self.memory = deque(maxlen=buffer_size)
        self.step_count = 0

    def act(self, state, epsilon=0.0):
        """Choose action using epsilon-greedy policy"""
        if random.random() > epsilon:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            # Sample random quantiles for action selection
            quantiles = torch.rand(1, self.n_quantiles).to(self.device)

            with torch.no_grad():
                quantile_values = self.q_network(state, quantiles)
                # Average over quantiles to get expected Q-values
                q_values = quantile_values.mean(dim=1)  # [1, action_size]
                action = q_values.argmax().item()
        else:
            action = random.choice(np.arange(self.action_size))

        return action

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """Train the network on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return

        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)

        # Sample random quantiles
        quantiles = torch.rand(self.batch_size, self.n_quantiles).to(self.device)
        next_quantiles = torch.rand(self.batch_size, self.n_quantiles).to(self.device)

        # Current quantile values
        current_quantiles = self.q_network(
            states, quantiles
        )  # [batch_size, n_quantiles, action_size]
        current_quantiles = current_quantiles.gather(
            2, actions.unsqueeze(1).unsqueeze(2).expand(-1, self.n_quantiles, 1)
        ).squeeze(2)

        # Next quantile values for target
        with torch.no_grad():
            # Get next actions using current network (Double DQN style)
            next_quantiles_current = self.q_network(next_states, next_quantiles)
            next_q_values = next_quantiles_current.mean(
                dim=1
            )  # [batch_size, action_size]
            next_actions = next_q_values.argmax(dim=1)  # [batch_size]

            # Get target quantile values
            next_quantiles_target = self.target_network(next_states, next_quantiles)
            next_quantiles_target = next_quantiles_target.gather(
                2,
                next_actions.unsqueeze(1).unsqueeze(2).expand(-1, self.n_quantiles, 1),
            ).squeeze(2)

            # Compute targets
            targets = rewards.unsqueeze(1) + (
                self.gamma * next_quantiles_target * ~dones.unsqueeze(1)
            )

        # ipdb.set_trace()
        # Compute quantile Huber loss
        td_errors = targets.unsqueeze(1) - current_quantiles.unsqueeze(
            2
        )  # [batch_size, n_quantiles, n_quantiles]
        huber_loss = F.smooth_l1_loss(
            current_quantiles.unsqueeze(2),
            targets.unsqueeze(1),
            reduction="none",
            beta=self.kappa,
        )
        quantile_loss = (
            torch.abs(quantiles.unsqueeze(2) - (td_errors < 0).float()) * huber_loss
        )
        loss = quantile_loss.mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def get_distribution(self, state, action, num_quantiles=100):
        """Get the return distribution for a state-action pair"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        quantiles = (
            torch.linspace(0.01, 0.99, num_quantiles).unsqueeze(0).to(self.device)
        )

        with torch.no_grad():
            quantile_values = self.q_network(state, quantiles)
            distribution = quantile_values[0, :, action].cpu().numpy()

        return distribution, np.linspace(0.01, 0.99, num_quantiles)


def train_iqn_lunarlander():
    """Train IQN on LunarLander environment"""
    env = gym.make("LunarLander-v3")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = IQNAgent(state_size, action_size)

    episodes = 1000
    scores = deque(maxlen=100)
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01

    for episode in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        total_reward = 0
        done = False

        while not done:
            action = agent.act(state, epsilon)
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated

            agent.remember(state, action, reward, next_state, done)
            agent.replay()

            state = next_state
            total_reward += reward

        scores.append(total_reward)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if episode % 100 == 0:
            print(
                f"Episode {episode}, Average Score: {np.mean(scores):.2f}, Epsilon: {epsilon:.3f}"
            )

    return agent, scores


# Example usage and analysis
if __name__ == "__main__":
    # Train the agent
    agent, scores = train_iqn_lunarlander()

    # Plot training progress
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(scores)
    plt.title("Training Scores")
    plt.xlabel("Episode")
    plt.ylabel("Score")

    # Analyze return distribution for a sample state
    env = gym.make("LunarLander-v3")
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]

    # Get distribution for each action
    plt.subplot(1, 2, 2)
    for action in range(env.action_space.n):
        distribution, quantiles = agent.get_distribution(state, action)
        plt.plot(quantiles, distribution, label=f"Action {action}")

    plt.title("Q-value Distributions")
    plt.xlabel("Quantile")
    plt.ylabel("Q-value")
    plt.legend()
    plt.tight_layout()
    plt.show()
