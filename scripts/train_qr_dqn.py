import torch
from porl.env.env import lunarLander
from porl.train.qr_dqn_trainer import QRDQNTrainer


def main():
    """
    Main function to initialize and run the QR-DQN training process.
    """
    # Initialize the LunarLander environment
    env, state_size, action_size = lunarLander()

    # Set the device for PyTorch computations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Instantiate the QRDQNTrainer
    trainer = QRDQNTrainer(
        state_size=state_size,
        action_size=action_size,
        network_hidden_sizes=[128, 128],  # Standard hidden layer sizes
        gamma=0.99,  # Discount factor
        epsilon=1.0,  # Initial exploration rate
        epsilon_min=0.01,  # Minimum exploration rate
        # epsilon_decay=0.995,  # Exploration rate decay factor
        epsilon_decay=0.999,  # Exploration rate decay factor
        # update_target_freq=100,  # Frequency of updating the target network (in episodes)
        update_target_freq=500,  # Frequency of updating the target network (in episodes)
        # batch_size=64,  # Mini-batch size for training
        # learning_rate=5e-4,  # Learning rate for the Adam optimizer
        learning_rate=1e-4,  # Learning rate for the Adam optimizer
        num_quantiles=51,  # Number of quantiles for QR-DQN
        kappa=1.0,  # Huber loss parameter for QR-DQN
        device=device,  # Device to run on (CPU or CUDA)
        log_dir="logs/qr_dqn",  # Directory for logs
    )

    # trainer = C51Trainer(
    #     state_size=state_size,
    #     action_size=action_size,
    #     gamma=0.99,
    #     epsilon=1.0,
    #     epsilon_min=0.1,
    #     epsilon_decay=0.95,
    #     update_target_freq=10,
    #     device=device,
    #     atom_size=51,
    #     # v_min=-10,
    #     # v_max=10,
    #     v_min=-300,
    #     v_max=300,
    #     log_dir="logs",
    # )

    # Start the online training process
    # num_episodes can be adjusted (e.g., 600 like in other scripts, or 1000)
    print(f"Starting training for {trainer.__class__.__name__} on LunarLander...")
    trainer.train_online(env, num_episodes=1000)
    print("Training complete.")


if __name__ == "__main__":
    main()
