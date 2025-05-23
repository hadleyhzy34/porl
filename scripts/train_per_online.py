import gymnasium as gym
from porl.train.dqn_per_trainer import PERTrainer
from porl.env.env import lunarLander
from porl.policy.collect_dataset import collect_dataset
from porl.net.q_network import QNetwork
import torch


def main():
    env, state_size, action_size = lunarLander()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = PERTrainer(
        state_size=state_size,
        action_size=action_size,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.95,
        update_target_freq=10,
        device=device,
        network=QNetwork,
        log_dir="logs",
    )

    # train online
    trainer.train_online(env)


if __name__ == "__main__":
    main()
