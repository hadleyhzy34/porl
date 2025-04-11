import gymnasium as gym
from porl.policy.dqn import ddqn_learn, dqn_learn
from porl.train.dqn_trainer import DQNTrainer
from porl.env.env import lunarLander
from porl.net.q_network import QNetwork
import torch


def main():
    env, state_size, action_size = lunarLander()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = DQNTrainer(
        state_size=state_size,
        action_size=action_size,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.95,
        update_target_freq=10,
        device=device,
        network=QNetwork,
        log_dir="logs",
    )
    trainer.train(num_episodes=1000, policy=ddqn_learn, env=env)


if __name__ == "__main__":
    main()
