import rospy
import statistics
from env.gazebo import Env

def evaluate_policy(agent, args):
    rospy.init_node(args.namespace)

    EPISODES = args.test_episodes

    env = Env(args.namespace,args.state_size,args.action_size)

    scores = []
    success = []
    steps = []

    for e in range(EPISODES):
        done = False
        state = env.reset()

        score = 0
        for t in range(agent.episode_step):
            action = agent.select_action(state)

            # execute actions and wait until next scan(state)
            next_state, reward, done, truncated, info = env.step(action)

            score += reward
            state = next_state

            if t == agent.episode_step - 1:
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
