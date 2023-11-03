import torch
import random
import string
import numpy as np
from torch.utils.data import DataLoader, Dataset
from agent.fasternet import FasterNet
from agent.value_functions import TwinV
from dataloader.dataloader import CustomDataset
import pdb
from matplotlib import pyplot as plt
from tqdm import tqdm
import statistics
import time
import argparse
from agent.policy import GaussianPolicy
from agent.por import POR
from torch.utils.tensorboard import SummaryWriter
from test import evaluate_policy

def train(args):
    torch.autograd.set_detect_anomaly(True)
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device(args.device)

    #summary writer session name
    res = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
    writer = SummaryWriter(f'./log/{res}')

    state_size = args.state_size
    action_size = args.action_size

    backbone = FasterNet(3,args.feature_dim)

    # policy = GaussianPolicy(obs_dim + obs_dim, act_dim, hidden_dim=1024, n_hidden=2)
    # policy to predict next state
    # goal_policy = GaussianPolicy(state_size,
    #                              state_size,
    #                              hidden_dim=args.hidden_dim,
    #                              n_hidden=args.n_hidden)
    #
    # # state value function
    # vf = TwinV(state_size,
    #            layer_norm=args.layer_norm,
    #            hidden_dim=args.hidden_dim,
    #            n_hidden=args.n_hidden)

    agent = POR(args,
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

    steps = 0

    for i in range(args.episodes):
        e_v_loss = []
        e_g_loss = []

        for data in (pbar := tqdm(train_dataloader)):
            # pdb.set_trace()
            # s,r,s',d,a
            observations = data[:,:362]
            rewards = data[:,362]
            next_observations = data[:,363:-3]
            dones = data[:,-3]
            v_loss, g_loss = agent.por_residual_update(observations,
                                                       next_observations,
                                                       rewards,
                                                       dones)
            e_v_loss.append(v_loss)
            e_g_loss.append(g_loss)

            pbar.set_description(f"episode:{i}||"
                                 f"v_loss: {statistics.fmean(e_v_loss):.4f}||"
                                 f"g_loss: {statistics.fmean(e_g_loss):.4f} "
                    )

            writer.add_scalar('v_loss',v_loss, steps)
            writer.add_scalar('g_loss',g_loss, steps)
            steps += 1

        # evaluate policy every 10 episodes
        if (i+1) % 10 == 0:
            mean_stp_length, mean_rew, mean_success_rate = evaluate_policy(agent,args)
            print(f"episodes: {i}||"
                  f"mean_step_length: {mean_stp_length}||"
                  f"mean_reward: {mean_rew}||"
                  f"mean_success_rate: {mean_success_rate}"
                  )

        # t = time.localtime()
        # current_time = time.strftime("%H_%M_%S", t)
        # torch.save(agent.state_dict(), Config.weight_folder+current_time+'.pt')
        # torch.save(agent.state_dict(), Config.weight_folder + str(i) + '.pth')

        # if i > 10:
            # agent.scheduler.step()
        # agent.scheduler.step()

    # plt.plot(np.arange(len(total_all_loss)),total_all_loss,label="total_loss")
    # plt.plot(np.arange(len(total_all_loss)),total_dist_loss,label="dist_loss")
    # plt.plot(np.arange(len(total_all_loss)),total_coll_loss,label="coll_loss")
    # plt.plot(np.arange(len(total_all_loss)),total_angl_loss,label="angle_loss")
    # # plt.plot(np.arange(len(total_all_loss)),total_step_loss,label="step_loss")
    # plt.legend()
    # plt.show()

    # torch.save(agent.state_dict(), Config.weight_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--namespace', type=str, default='tb3')
    parser.add_argument('--state_size', type=int, default=362)
    parser.add_argument('--action_size', type=int, default=2)
    parser.add_argument('--episodes', type=int, default=5000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--replay_buffer_size', type=int, default=10_000)
    parser.add_argument('--episode_step', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--rank_update_interval', type=int, default=200)
    parser.add_argument('--learning_starts', type=int, default=1000)
    parser.add_argument('--target_update_interval', type=int, default=1)
    parser.add_argument('--total_timesteps', type=int, default=100_000)
    parser.add_argument('--hidden_dim', type=int, default=1024)
    parser.add_argument('--n_hidden', type=int, default=2)
    parser.add_argument("--normalize", action='store_true')
    parser.add_argument("--layer_norm", action='store_true')
    parser.add_argument("--train_steps", type=int, default=1_000)
    parser.add_argument("--feature_dim", type=int, default=256)

    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.9)
    parser.add_argument('--value_lr', type=float, default=1e-4)
    parser.add_argument('--policy_lr', type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=10.0)
    args = parser.parse_args()
    train(args)
