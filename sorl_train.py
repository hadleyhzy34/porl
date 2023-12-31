from email import policy
from email.policy import default
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from agent.value_functions import TwinV
from dataloader.dataloader import CustomDataset
import pdb
from matplotlib import pyplot as plt
from tqdm import tqdm
import statistics
import time
import argparse
from agent.policy import GaussianPolicy
from agent.sorl import SORL
from agent.fasternet import FasterNet
from test import evaluate_policy

def train(args):
    torch.autograd.set_detect_anomaly(True)
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device(args.device)

    state_size = args.state_size
    action_size = args.action_size

    backbone = FasterNet(3,args.feature_dim)

    agent = SORL(args,
                 backbone=backbone,
                 max_steps=args.train_steps,
                 tau=args.tau,
                 alpha=args.alpha,
                 discount=args.discount,
                 value_lr=args.value_lr,
                 policy_lr=args.policy_lr,
                 device=device
                )

    data = CustomDataset(device = device)

    train_dataloader = DataLoader(data,
                                  batch_size=args.batch_size,
                                  shuffle=True)

    for i in range(args.episodes):
        e_v_loss = []
        e_g_loss = []

        for data in (pbar := tqdm(train_dataloader)):
            # pdb.set_trace()
            observations = data[:,:362]
            rewards = data[:,362]
            next_observations = data[:,363:-3]
            dones = data[:,-3]
            actions = data[:,-2:]
            v_loss, g_loss = agent.update(observations,
                                          actions,
                                          rewards,
                                          next_observations,
                                          dones)
            e_v_loss.append(v_loss)
            e_g_loss.append(g_loss)

            pbar.set_description(f"episode:{i}||"
                                 f"v_loss: {statistics.fmean(e_v_loss):.4f}||"
                                 f"g_loss: {statistics.fmean(e_g_loss):.4f} "
                    )

        # evaluate policy every 10 episodes
        if args.evaluate is True and (i+1) % 2 == 0:
            torch.save(agent.state_dict(), f'weights/model_{i}.pt')
            mean_stp_length, mean_rew, mean_success_rate = evaluate_policy(agent, args)
            print(f"eval_step: {i}||"
                  f"mean_step_length: {mean_stp_length}||"
                  f"mean_reward: {mean_rew}||"
                  f"mean_success_rate: {mean_success_rate}"
                  )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--namespace', type=str, default='tb3')
    parser.add_argument('--state_size', type=int, default=362)
    parser.add_argument('--test_episodes', type=int, default=10)
    parser.add_argument('--episode_step', type=int, default=500)
    parser.add_argument('--action_size', type=int, default=2)
    parser.add_argument('--episodes', type=int, default=5000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--n_hidden', type=int, default=2)
    parser.add_argument("--normalize", action='store_true')
    parser.add_argument("--layer_norm", action='store_true')
    parser.add_argument("--train_steps", type=int, default=1_000)
    parser.add_argument('--feature_dim', type=int, default=256)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.9)
    parser.add_argument('--value_lr', type=float, default=1e-4)
    parser.add_argument('--policy_lr', type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=10.0)
    parser.add_argument('--evaluate', action="store_true")
    args = parser.parse_args()
    train(args)
