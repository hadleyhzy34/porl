import gymnasium as gym
from porl.policy.dqn import ddqn_learn, dqn_learn
from porl.train.ddqn_trainer import DDQNTrainer
from porl.env.env import lunarLander
from porl.net.q_network import QNetwork, DuelingQNetwork
import torch


def main():
    env, state_size, action_size = lunarLander()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dueling network + double dqn
    trainer = DDQNTrainer(
        state_size=state_size,
        action_size=action_size,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.95,
        update_target_freq=10,
        device=device,
        network=DuelingQNetwork,
        log_dir="logs",
    )
    trainer.train_online(env)


if __name__ == "__main__":
    main()
