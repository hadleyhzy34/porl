from platform import node
import rospy
import random
import argparse
import numpy as np
import pdb
import torch
import torch.nn as nn
# from env.env import Env
from env.gazebo import Env
import torch.multiprocessing as mp
from multiprocessing import Lock
from concurrent.futures import process

def collect(namespace, rank, args):
    # pdb.set_trace()
    node_name = f"{namespace}_{rank}"
    rospy.init_node(node_name)

    EPISODES = args.episodes

    env = Env(namespace,
              args.state_size,
              args.action_size,
              rank)

    action_bound = np.array([0.15/2,1.5])

    total_data = []
    file_index = 0

    for e in range(EPISODES):
        done = False
        state = env.reset()

        score = 0
        for t in range(args.episode_step):
            action = (np.random.random((2,)) * 2 + np.array([0,-1.])) * action_bound

            # execute actions and wait until next scan(state)
            next_state, reward, done, truncated, info = env.collect_step(action)

            score += reward
            state = next_state

            data = {'state': state,
                    'next_state': next_state,
                    'reward':reward,
                    'done':done,
                    'truncated':truncated}

            total_data.append(data)
            if len(total_data) == 100:
                torch.save(total_data,f'checkpoint/dataset_{rank}_{file_index}.pt')
                total_data = []
                file_index += 1

            if done:
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--namespace', type=str, default='tb3')
    parser.add_argument('--state_size', type=int, default=256)
    parser.add_argument('--action_size', type=int, default=2)
    parser.add_argument('--episodes', type=int, default=5000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--replay_buffer_size', type=int, default=5000)
    parser.add_argument('--episode_step', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--update_step', type=int, default=500)
    parser.add_argument('--num_processes', type=int, default=4)
    args = parser.parse_args()

    # rospy.init_node('collect')
    processes = []
    for rank in range(args.num_processes):
        p = mp.Process(target=collect, args=(args.namespace+'_'+str(rank), rank, args,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
