import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ipdb
from typing import List
from porl.train.dqn_trainer import DQNTrainer
from porl.net.iqn_network import IQNNetwork
from porl.buffer.replaybuffer import ReplayBuffer


class IQNTrainer(DQNTrainer):
    """
    Implicit Quantile Network (IQN) Trainer.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        gamma: float,
        epsilon: float,
        epsilon_min: float,
        epsilon_decay: float,
        update_target_freq: int,
        device: torch.device,
        network_hidden_sizes: List[int] = [128, 128],
        learning_rate: float = 5e-4,
        buffer_size: int = 100000,
        batch_size: int = 64,
        kappa: float = 1.0,
        embedding_dim: int = 64,
        num_quantiles_k: int = 8,
        num_quantiles_n_policy: int = 32,
        num_quantiles_n_prime_loss: int = 8,
        num_quantiles_n_double_prime_loss: int = 8,
        dueling_network: bool = True,
        log_dir: str = "logs",
    ):
        super().__init__(
            state_size=state_size,
            action_size=action_size,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            learning_rate=learning_rate,
            update_target_freq=update_target_freq,
            device=device,
            network=None,
            log_dir=log_dir,
            # replay_buffer=ReplayBuffer(buffer_size, (state_size,), device, batch_size=batch_size),
        )
        self.batch_size = batch_size
        self.kappa = kappa
        self.embedding_dim = embedding_dim
        self.num_quantiles_k = num_quantiles_k
        self.num_quantiles_n_policy = num_quantiles_n_policy
        self.num_quantiles_n_prime_loss = num_quantiles_n_prime_loss
        self.num_quantiles_n_double_prime_loss = num_quantiles_n_double_prime_loss
        self.dueling_network = dueling_network

        self.q_network = IQNNetwork(
            state_size,
            action_size,
            embedding_dim,
            num_quantiles_k,
            self.dueling_network,
        )

        self.target_network = IQNNetwork(
            state_size,
            action_size,
            embedding_dim,
            num_quantiles_k,
            dueling_network,
        ).to(device)

        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)

    def select_action(self, state: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        taus = torch.rand(1, self.num_quantiles_n_policy).to(self.device)
        with torch.no_grad():
            q_values_per_tau = self.q_network.get_q_values(
                state_tensor, taus
            )  # (1, N, action_dim)
            q_values = q_values_per_tau.mean(dim=1)  # (1, action_dim)
        return q_values.argmax(dim=1).item()

    def learn(self) -> float:
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        device = self.device

        # 1. Sample taus for current quantile estimation (N')
        taus_prime = torch.rand(states.shape[0], self.num_quantiles_n_prime_loss, device=device)
        # 2. Get current quantiles: (B, N', action_dim)
        current_quantiles_all_actions = self.q_network.get_q_values(states, taus_prime)

        # 3. Gather quantiles for the taken actions
        # actions: (B,) -> (B, 1, 1), then expand to (B, N', 1)
        actions_expanded = actions.view(-1, 1, 1).expand(-1, self.num_quantiles_n_prime_loss, 1)
        current_sa_quantiles = current_quantiles_all_actions.gather(2, actions_expanded).squeeze(2)  # (B, N')

        with torch.no_grad():
            # 4. Sample taus for target quantile estimation (N'')
            taus_double_prime = torch.rand(states.shape[0], self.num_quantiles_n_double_prime_loss, device=device)
            # 5. Next state Q-values for action selection (Double DQN)
            next_q_values_per_tau = self.q_network.get_q_values(next_states, taus_double_prime)  # (B, N'', action_dim)
            next_q_values = next_q_values_per_tau.mean(dim=1)  # (B, action_dim)
            next_actions = torch.argmax(next_q_values, dim=1, keepdim=True)  # (B, 1)

            # 6. Get target quantiles for next state and selected action
            target_quantiles_all_actions = self.target_network.get_q_values(next_states, taus_double_prime)  # (B, N'', action_dim)
            next_actions_expanded = next_actions.view(-1, 1, 1).expand(-1, self.num_quantiles_n_double_prime_loss, 1)
            target_sa_quantiles = target_quantiles_all_actions.gather(2, next_actions_expanded).squeeze(2)  # (B, N'')

            # 7. Bellman target: (B, N'')
            td_target_quantiles = rewards.unsqueeze(1) + self.gamma * target_sa_quantiles * (1 - dones.unsqueeze(1))

        # 8. Quantile regression loss
        # current_sa_quantiles: (B, N'), td_target_quantiles: (B, N'')
        # td_errors: (B, N', N'')
        td_errors = td_target_quantiles.unsqueeze(1) - current_sa_quantiles.unsqueeze(2)
        loss = self.quantile_huber_loss(td_errors, taus_prime)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()
        return loss.item()

    def quantile_huber_loss(self, td_errors, taus):
        # td_errors: (B, N', N'')
        abs_td_errors = torch.abs(td_errors)
        huber_loss = torch.where(
            abs_td_errors <= self.kappa,
            0.5 * td_errors.pow(2),
            self.kappa * (abs_td_errors - 0.5 * self.kappa),
        )
        indicator = (td_errors < 0).float()
        taus = taus.unsqueeze(-1)  # (B, N', 1)
        quantile_regression_factor = torch.abs(taus - indicator)
        loss = (quantile_regression_factor * huber_loss).mean(dim=2).mean(dim=1).mean()
        return loss
