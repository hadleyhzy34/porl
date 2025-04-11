import torch
import torch.nn as nn
from util.util import mlp


class TwinQ(nn.Module):
    def __init__(agent, state_dim, action_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        agent.q1 = mlp(dims, squeeze_output=True)
        agent.q2 = mlp(dims, squeeze_output=True)

    def both(agent, state, action):
        sa = torch.cat([state, action], 1)
        return agent.q1(sa), agent.q2(sa)

    def forward(agent, state, action):
        return torch.min(*agent.both(state, action))


class ValueFunction(nn.Module):
    def __init__(agent, state_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        agent.v = mlp(dims, squeeze_output=True)

    def forward(agent, state):
        return agent.v(state)


class TwinV(nn.Module):
    def __init__(agent, state_dim, layer_norm=False, hidden_dim=256, n_hidden=2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        agent.v1 = mlp(dims, layer_norm=layer_norm, squeeze_output=True)
        agent.v2 = mlp(dims, layer_norm=layer_norm, squeeze_output=True)

    def both(agent, state):
        return agent.v1(state), agent.v2(state)

    def forward(agent, state):
        return torch.min(*agent.both(state))
