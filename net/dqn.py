import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_sizes=[64, 128, 64]):
        super(QNetwork, self).__init__()
        layers = []
        cur_size = state_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(cur_size, hidden_size))
            layers.append(nn.ReLU())
            cur_size = hidden_size
        layers.append(nn.Linear(cur_size, action_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
