from email.policy import default
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from agent.value_functions import TwinV
from agent.agent import Agent
from dataloader.dataloader import CustomDataset
import pdb
from matplotlib import pyplot as plt
from tqdm import tqdm
import statistics
import time
import argparse
from agent.policy import GaussianPolicy
from agent.por import POR

def train(args):
    torch.autograd.set_detect_anomaly(True)
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device(args.device)

    state_size = args.state_size
    action_size = args.action_size

    # policy = GaussianPolicy(obs_dim + obs_dim, act_dim, hidden_dim=1024, n_hidden=2)
    # policy to predict next state
    goal_policy = GaussianPolicy(state_size,
                                 state_size,
                                 hidden_dim=args.hidden_dim,
                                 n_hidden=args.n_hidden)

    # state value function
    vf = TwinV(state_size,
               layer_norm=args.layer_norm,
               hidden_dim=args.hidden_dim,
               n_hidden=args.n_hidden)

    agent = POR(
        vf=vf,
        goal_policy=goal_policy,
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

    total_all_loss = []
    total_dist_loss = []
    total_coll_loss = []
    total_angl_loss = []
    total_bspline_loss = []
    # total_step_loss = []

    for i in range(args.episodes):
        # print(f"current learning rate: {agent.optimizer.param_groups[0]['lr']}")
        # total_loss = []
        # dist_loss = []
        # coll_loss = []
        # angl_loss = []
        # bspline_loss = []
        # step_loss = []
        e_v_loss = []
        e_g_loss = []

        for data in (pbar := tqdm(train_dataloader)):
            # pdb.set_trace()
            observations = data[:,:362]
            rewards = data[:,362]
            next_observations = data[:,363:-1]
            dones = data[:,-1]
            v_loss, g_loss = agent.por_residual_update(observations,next_observations,rewards,dones)
            # data = data.to(Config.Train.device).float()
            # traj = agent.path_planning(data)
            #
            # # loss, loss_dist, loss_col, loss_angle, loss_bspline = agent.learn(traj, data, train=True)
            # loss, loss_dist, loss_col, loss_angle = agent.learn(traj, data, train=True)
            # total_loss.append(loss)
            # dist_loss.append(loss_dist)
            # coll_loss.append(loss_col)
            # angl_loss.append(loss_angle)
            # bspline_loss.append(loss_bspline)
            # step_loss.append(loss_step)
            e_v_loss.append(v_loss)
            e_g_loss.append(g_loss)

            pbar.set_description(f"episode:{i}||"
                                 f"v_loss: {statistics.fmean(e_v_loss):.4f}||"
                                 f"g_loss: {statistics.fmean(e_g_loss):.4f} "
                    )
            # pbar.set_description("loss: %.4f, dist: %.4f, coll: %.4f, angle: %.4f, bspline: %.4f"%(statistics.fmean(total_loss),statistics.fmean(dist_loss),statistics.fmean(coll_loss),statistics.fmean(angl_loss),statistics.fmean(bspline_loss)))
            # pbar.refresh()

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
    parser.add_argument('--action_size', type=int, default=5)
    parser.add_argument('--episodes', type=int, default=5000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=512)
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

    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.9)
    parser.add_argument('--value_lr', type=float, default=1e-4)
    parser.add_argument('--policy_lr', type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=10.0)
    args = parser.parse_args()
    train(args)
