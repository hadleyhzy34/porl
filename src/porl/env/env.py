import gymnasium as gym
import pdb


def lunarLander():
    env = gym.make("LunarLander-v3")
    # pdb.set_trace()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    return env, state_size, action_size
