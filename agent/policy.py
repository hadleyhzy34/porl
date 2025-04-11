import torch
import torch.nn as nn
import pdb
from torch.distributions import MultivariateNormal, Normal
from util.util import mlp


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class GaussianPolicy(nn.Module):
    def __init__(agent, obs_dim, act_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        agent.net = mlp([obs_dim, *([hidden_dim] * n_hidden), act_dim])
        agent.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))

    def forward(agent, obs):
        # pdb.set_trace()
        mean = agent.net(obs)
        std = torch.exp(agent.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        scale_tril = torch.diag(std)
        return MultivariateNormal(mean, scale_tril=scale_tril)
        # if mean.ndim > 1:
        #     batch_size = len(obs)
        #     return MultivariateNormal(mean, scale_tril=scale_tril.repeat(batch_size, 1, 1))
        # else:
        #     return MultivariateNormal(mean, scale_tril=scale_tril)

    def act(agent, obs, deterministic=False, enable_grad=False):
        with torch.set_grad_enabled(enable_grad):
            dist = agent(obs)
            return dist.mean if deterministic else dist.sample()

class BoundedGaussianPolicy(nn.Module):
    def __init__(agent, obs_dim, act_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        agent.net = mlp([obs_dim, *([hidden_dim] * n_hidden), act_dim],
                       output_activation=nn.Tanh)
        agent.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))

    def forward(agent, obs):
        # pdb.set_trace()
        mean = agent.net(obs)
        if mean.max() > 1. or mean.min() < -1.:
            pdb.set_trace()
        std = torch.exp(agent.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        scale_tril = torch.diag(std)
        return MultivariateNormal(mean, scale_tril=scale_tril)
        # if mean.ndim > 1:
        #     batch_size = len(obs)
        #     return MultivariateNormal(mean, scale_tril=scale_tril.repeat(batch_size, 1, 1))
        # else:
        #     return MultivariateNormal(mean, scale_tril=scale_tril)

    def act(agent, obs, deterministic=False, enable_grad=False):
        with torch.set_grad_enabled(enable_grad):
            dist = agent(obs)
            return dist.mean if deterministic else dist.sample()


class DeterministicPolicy(nn.Module):
    def __init__(agent, obs_dim, act_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        agent.net = mlp([obs_dim, *([hidden_dim] * n_hidden), act_dim],
                       output_activation=nn.Tanh)

    def forward(agent, obs):
        return agent.net(obs)

    def act(agent, obs, deterministic=False, enable_grad=False):
        with torch.set_grad_enabled(enable_grad):
            return agent(obs)
