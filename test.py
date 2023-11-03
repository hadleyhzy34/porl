import rospy
import numpy as np
import pdb
import statistics
import torch
from env.gazebo import Env

def evaluate_policy(agent, args):
    rospy.init_node(args.namespace)

    EPISODES = args.test_episodes

    env = Env(args.namespace,args.state_size,args.action_size)

    scores = []
    success = []
    steps = []

    action_bound = np.array([0.15/2,1.5])

    for e in range(EPISODES):
        done = False
        state, _ = env.reset()

        score = 0
        for t in range(args.episode_step):
            # pdb.set_trace()
            state = torch.from_numpy(state).to(torch.float).to(args.device)
            action = agent.select_action(state[None,:])[0]
            action = (action + np.array([1.,0.])) * action_bound

            assert action[0] >= 0 and action[0] <= 0.15, f"linear velocity is not in range: {action[0]}"
            assert action[1] >= -1.5 and action[1] <= 1.5, f"angular velocity is not in range: {action[1]}"

            # execute actions and wait until next scan(state)
            next_state, reward, done, truncated, info = env.step(action)

            score += reward
            state = next_state

            if t == args.episode_step - 1:
                done = True

            if done:
                scores.append(score)
                steps.append(t)
                if info['status'] == 'goal':
                    success.append(1.)
                else:
                    success.append(0.)
                break

    return statistics.fmean(steps), statistics.fmean(scores), statistics.fmean(success)
