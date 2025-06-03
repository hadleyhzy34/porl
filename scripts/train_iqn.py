import gymnasium as gym # Using Gymnasium instead of OpenAI Gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import os
import sys

# Add project root to Python path to allow imports from src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.porl.policy.iqn import IQNPolicy
from src.porl.train.iqn_trainer import IQNTrainer
from src.porl.buffer.replaybuffer import ReplayBuffer # Assuming a standard replay buffer

def main(config):
    # Environment
    env = gym.make(config['env_name'])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() and config['use_gpu'] else "cpu")
    print(f"Using device: {device}")

    # Seed for reproducibility
    if config['seed'] is not None:
        torch.manual_seed(config['seed'])
        np.random.seed(config['seed'])
        random.seed(config['seed'])
        # env.seed(config['seed']) # For older gym versions
        env.action_space.seed(config['seed'])
        # env.observation_space.seed(config['seed']) # Not standard for obs space

    # Policy
    policy = IQNPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        embedding_dim=config['embedding_dim'],
        num_quantiles_network_output=config['num_quantiles_k'],
        num_quantiles_policy_sample=config['num_quantiles_n_policy'],
        epsilon_start=config['epsilon_start'],
        epsilon_end=config['epsilon_end'],
        epsilon_decay=config['epsilon_decay_steps'],
        dueling_network=config['dueling_network'],
        device=device
    )

    # Replay Buffer
    replay_buffer = ReplayBuffer(
        buffer_size=config['buffer_size'],
        batch_size=config['batch_size'],
        device=device,
        state_dim=state_dim, # Required by our ReplayBuffer
        action_dim=1 # Assuming action is a single integer
    )

    # Trainer
    trainer = IQNTrainer(
        policy=policy,
        learning_rate=config['learning_rate'],
        gamma=config['gamma'],
        batch_size=config['batch_size'],
        target_update_frequency=config['target_update_freq_steps'],
        kappa=config['huber_loss_kappa'],
        num_quantiles_loss_samples_N_prime=config['num_quantiles_n_prime_loss'],
        num_quantiles_loss_samples_N_double_prime=config['num_quantiles_n_double_prime_loss'],
        device=device
    )

    scores = []
    losses = []
    scores_window = deque(maxlen=100)
    epsilons = []

    print(f"Starting training on {config['env_name']} for {config['num_episodes']} episodes...")

    for i_episode in range(1, config['num_episodes'] + 1):
        state, info = env.reset(seed=config['seed'] + i_episode if config['seed'] is not None else None)
        current_score = 0
        done = False
        truncated = False # For Gymnasium
        episode_losses = []

        for t in range(config['max_timesteps_per_episode']):
            action = policy.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)

            # Store experience
            replay_buffer.push(state, action, reward, next_state, float(done)) # Ensure done is float

            state = next_state
            current_score += reward

            # Train if buffer has enough samples
            if len(replay_buffer) >= config['batch_size'] and t % config['train_frequency_steps'] == 0:
                experiences = replay_buffer.sample()
                loss = trainer.train_step(experiences)
                episode_losses.append(loss)

            if done or truncated:
                break

        scores_window.append(current_score)
        scores.append(current_score)
        if episode_losses:
            losses.append(np.mean(episode_losses))
        epsilons.append(policy.epsilon)

        if i_episode % config['print_every_episodes'] == 0:
            print(f"Episode {i_episode}	Average Score (last 100): {np.mean(scores_window):.2f}	"
                  f"Current Score: {current_score:.2f}	Epsilon: {policy.epsilon:.3f}	"
                  f"Avg Loss (episode): {np.mean(episode_losses) if episode_losses else 0:.4f}")

        if np.mean(scores_window) >= config['solve_score'] and len(scores_window) >= 100:
            print(f"
Environment solved in {i_episode} episodes!	Average Score: {np.mean(scores_window):.2f}")
            # torch.save(policy.network.state_dict(), f"iqn_{config['env_name']}_solved.pth") # Save model
            break

    env.close()

    # Plotting results
    if config['plot_results']:
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.plot(scores)
        plt.title('Episode Scores')
        plt.xlabel('Episode')
        plt.ylabel('Score')

        plt.subplot(2, 2, 2)
        plt.plot(losses)
        plt.title('Training Loss per Episode (Avg)')
        plt.xlabel('Episode')
        plt.ylabel('Loss')

        plt.subplot(2, 2, 3)
        avg_scores = [np.mean(scores[max(0, i-100):i+1]) for i in range(len(scores))]
        plt.plot(avg_scores)
        plt.title('Average Score (Rolling 100 episodes)')
        plt.xlabel('Episode')
        plt.ylabel('Average Score')

        plt.subplot(2, 2, 4)
        plt.plot(epsilons)
        plt.title('Epsilon Decay')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')

        plt.tight_layout()
        plt.savefig(f"iqn_{config['env_name']}_training_plots.png")
        print(f"Training plots saved to iqn_{config['env_name']}_training_plots.png")
        # plt.show()


if __name__ == '__main__':
    config = {
        'env_name': 'LunarLander-v2', # LunarLander-v3 does not exist, v2 is standard
        'num_episodes': 2000,
        'max_timesteps_per_episode': 1000,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay_steps': 30000, # Total steps for decay
        'buffer_size': 100000,
        'batch_size': 64,
        'gamma': 0.99,
        'learning_rate': 5e-4, # Standard for Adam
        'target_update_freq_steps': 100, # Update target net every N training steps
        'train_frequency_steps': 4, # Train policy every N environment steps
        'huber_loss_kappa': 1.0,
        'embedding_dim': 64, # For IQN cosine embedding
        'num_quantiles_k': 8,       # K: Number of quantiles network outputs
        'num_quantiles_n_policy': 32, # N: Number of taus sampled for policy evaluation
        'num_quantiles_n_prime_loss': 8,  # N': Number of taus for current quantiles in loss
        'num_quantiles_n_double_prime_loss': 8, # N'': Number of taus for target quantiles in loss
        'dueling_network': True,
        'use_gpu': True,
        'seed': 42, # For reproducibility
        'print_every_episodes': 10,
        'solve_score': 200.0, # Score to consider LunarLander-v2 solved
        'plot_results': True,
    }
    main(config)
