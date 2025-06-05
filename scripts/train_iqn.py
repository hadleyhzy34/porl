import torch
from porl.env.env import lunarLander
from porl.train.iqn_trainer import IQNTrainer

def main():
    env, state_size, action_size = lunarLander()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = IQNTrainer(
        state_size=state_size,
        action_size=action_size,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        update_target_freq=100,
        device=device,
        network_hidden_sizes=[128, 128],
        learning_rate=5e-4,
        buffer_size=100000,
        batch_size=64,
        kappa=1.0,
        embedding_dim=64,
        num_quantiles_k=8,
        num_quantiles_n_policy=32,
        num_quantiles_n_prime_loss=8,
        num_quantiles_n_double_prime_loss=8,
        dueling_network=True,
        log_dir="logs/iqn",
    )
    trainer.train_online(env, num_episodes=1000, max_steps=1000)

if __name__ == "__main__":
    main()