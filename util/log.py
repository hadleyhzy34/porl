import numpy as np
import pdb
from stable_baselines3.common.callbacks import BaseCallback

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(agent, verbose=0):
        super().__init__(verbose)

    def _on_step(agent) -> bool:
        # # Log scalar value (here a random variable)
        # value = np.random.random()
        # agent.logger.record("random_value", value)
        #
        # if agent.num_timesteps % 10 == 0:
        #     # agent.logger.record("reward", agent.locals['rewards'][0])
        #     agent.logger.dump(agent.locals['rewards'][0])
        # pdb.set_trace()
        # agent.logger.record("reward", agent.locals['rewards'][0])

        if agent.locals['infos'][0]['status'] == 'goal':
            agent.logger.record("status", 1.0)
        elif agent.locals['infos'][0]['status'] == 'hit':
            agent.logger.record("status", -1.)
        else:
            agent.logger.record("status", 0.)
        return True

    # def _on_rollout_end(agent) -> None:
    #     # pdb.set_trace()
    #     if agent.locals['dones'][0]:
    #         agent.logger.record("status", 1.0)
    #     else:
    #         if agent.locals['infos'][0]['TimeLimit.truncated']:
    #             agent.logger.record("status", 0.0)
    #         else:
    #             agent.logger.record("status", -1.)
