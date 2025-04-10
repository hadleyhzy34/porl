import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from collections import deque
import random
import pdb


def epsilon_greedy_policy(state, epsilon, q_network, action_size, device):
    if np.random.rand() < epsilon:
        return np.random.randint(action_size)  # Random action (explore)
    else:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)  # Add batch dimension
        with torch.no_grad():
            q_values = q_network(state)  # Get Q-values
        return q_values.argmax().item()  # Best action (exploit)


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_sizes=[64, 128, 64]):
        super(QNetwork, self).__init__()
        layers = []
        cur_size = state_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(cur_size, hidden_size))
            layers.append(nn.ReLU())
            cur_size = hidden_size
        layers.append(nn.Linear(cur_size, action_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


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


# class ReplayBuffer:
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


def runner():
    # Hyperparameters
    num_episodes = 500  # More episodes due to increased complexity
    max_steps = 1000  # Max steps per episode in LunarLander-v2
    gamma = 0.99  # Discount factor
    epsilon = 1.0  # Initial epsilon
    epsilon_min = 0.01  # Minimum epsilon
    epsilon_decay = 0.995  # Slower decay for more exploration
    update_target_freq = 10  # Update target network every 10 episodes

    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Initialize the environment
    env = gym.make("LunarLander-v3")
    state_size = env.observation_space.shape[0]  # 8 for LunarLander-v2
    action_size = env.action_space.n  # 4 for LunarLander-v2

    # Device configuration (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the replay buffer
    buffer_capacity = 100000  # Larger capacity for longer episodes
    batch_size = 128
    replay_buffer = ReplayBuffer(buffer_capacity, (state_size,), device)

    # Create the Q-network and target network
    q_network = QNetwork(state_size, action_size).to(device)
    target_network = QNetwork(state_size, action_size).to(device)
    target_network.load_state_dict(q_network.state_dict())  # Copy weights
    target_network.eval()  # Set target network to evaluation mode

    # Define optimizer
    optimizer = optim.Adam(q_network.parameters(), lr=0.0005)

    # Training loop
    rewards_history = []

    for episode in range(num_episodes):
        # pdb.set_trace()
        state, _ = env.reset()  # Reset environment, state is an 8D vector
        episode_reward = 0

        for step in range(max_steps):
            # Choose action using epsilon-greedy policy
            action = epsilon_greedy_policy(
                state, epsilon, q_network, action_size, device
            )

            # Take action and observe next state, reward, and done flag
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated  # Consider episode done if truncated

            # Store experience in replay buffer
            replay_buffer.push(state, action, reward, next_state, done)

            # Update state and episode reward
            state = next_state
            episode_reward += reward

            # Train the Q-network if we have enough experiences
            if len(replay_buffer) >= batch_size:
                # pdb.set_trace()

                # Sample directly as tensors
                states, actions, rewards, next_states, dones = replay_buffer.sample(
                    batch_size
                )

                # # Sample a batch of experiences
                # batch = replay_buffer.sample(batch_size)
                # states, actions, rewards, next_states, dones = zip(*batch)
                #
                # # Convert to PyTorch tensors
                # states = torch.FloatTensor(states).to(device)
                # actions = torch.LongTensor(actions).to(device)
                # rewards = torch.FloatTensor(rewards).to(device)
                # next_states = torch.FloatTensor(next_states).to(device)
                # dones = torch.FloatTensor(dones).to(device)

                # Compute target Q-values using the target network
                with torch.no_grad():
                    next_q_values = target_network(next_states)
                    max_next_q_values = next_q_values.max(dim=1)[0]
                    targets = rewards + gamma * max_next_q_values * (1 - dones)

                # Compute current Q-values and loss
                q_values = q_network(states)
                q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
                loss = F.mse_loss(q_values, targets)

                # Optimize the Q-network
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        # Update epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Update target network
        if episode % update_target_freq == 0:
            target_network.load_state_dict(q_network.state_dict())

        # Log episode reward
        rewards_history.append(episode_reward)
        if episode % 10 == 0:
            print(
                f"Episode {episode}, Reward: {episode_reward:.2f}, Epsilon: {epsilon:.3f}"
            )

    # Evaluation loop
    env = gym.make("LunarLander-v3", render_mode="human")
    num_eval_episodes = 5
    eval_epsilon = 0.05

    for episode in range(num_eval_episodes):
        state, _ = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = epsilon_greedy_policy(
                state, eval_epsilon, q_network, action_size, device
            )
            state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            if done or truncated:
                break

        print(f"Evaluation Episode {episode}, Reward: {episode_reward:.2f}")

    env.close()


if __name__ == "__main__":
    runner()
