import gymnasium as gym
import numpy as np
import pdb

if __name__ == "__main__":
    # pdb.set_trace()
    # Initialize environment
    env = gym.make("Taxi-v3")

    # Hyperparameters
    alpha = 0.1  # Learning rate
    gamma = 0.99  # Discount factor
    epsilon = 1.0  # Exploration rate
    epsilon_decay = 0.995
    epsilon_min = 0.01
    episodes = 2000

    # Q-table initialization (size: states x actions)
    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    # Training loop
    for episode in range(episodes):
        state = env.reset()[0]
        done = False

        while not done:
            # Epsilon-greedy action selection
            if np.random.uniform() < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(q_table[state])

            # Take action and observe next state and reward
            next_state, reward, done, _, _ = env.step(action)

            # Update Q-value
            q_table[state, action] += alpha * (
                reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
            )

            state = next_state

        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Evaluation
    state = env.reset()[0]
    total_reward = 0
    done = False
    eva_episodes = 10

    # Evaluation loop
    for episode in range(eva_episodes):
        while not done:
            action = np.argmax(q_table[state])
            state, reward, done, _, _ = env.step(action)
            total_reward += reward

    print(f"Total reward after training: {total_reward / eva_episodes}")
