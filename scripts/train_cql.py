import gymnasium as gym
from porl.net import q_network
from porl.net.behavior_policy import BehaviorPolicy
from porl.policy.dqn import ddqn_learn, dqn_learn
from porl.policy.collect_dataset import collect_dataset

# from porl.train.bcq_trainer import BCQTrainer
from porl.train.cql_trainer import CQLTrainer
from porl.env.env import lunarLander
from porl.net.q_network import DuelingQNetwork, QNetwork
import torch


def main():
    env, state_size, action_size = lunarLander()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = CQLTrainer(
        state_size=state_size,
        action_size=action_size,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.99,
        update_target_freq=10,
        device=device,
        network=QNetwork,
        log_dir="logs",
    )

    # collect dataset
    collect_dataset(env, trainer)

    #
    # train cql learning policy
    trainer.train_offline()


if __name__ == "__main__":
    main()
