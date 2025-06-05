import torch
import torch.nn.functional as F
import numpy as np
from typing import List
import pdb
import ipdb

from porl.train.dqn_trainer import DQNTrainer
from porl.net.qr_dqn_network import QRNetwork


class QRDQNTrainer(DQNTrainer):
    """
    Quantile Regression Deep Q-Network (QR-DQN) Trainer.
    This trainer implements the learning algorithm for QR-DQN, which models
    the distribution of returns for each action using a set of quantiles.
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
        num_quantiles: int = 51,
        kappa: float = 1.0,
        learning_rate: float = 5e-4,  # Added learning_rate to match DQNTrainer's optimizer
        log_dir: str = "logs",
    ):
        """
        Initializes the QRDQNTrainer.

        Args:
            state_size (int): Dimensionality of the state space.
            action_size (int): Number of possible actions.
            gamma (float): Discount factor for future rewards.
            epsilon (float): Initial exploration rate for epsilon-greedy policy.
            epsilon_min (float): Minimum exploration rate.
            epsilon_decay (float): Decay rate for exploration rate.
            update_target_freq (int): Frequency (in episodes) for updating the target network.
            device (torch.device): Device to run the computations on (e.g., 'cpu', 'cuda').
            network_hidden_sizes (List[int], optional): Sizes of hidden layers for the QRNetwork.
                                                       Defaults to [128, 128].
            num_quantiles (int, optional): Number of quantiles to predict. Defaults to 51.
            kappa (float, optional): Huber loss parameter. Defaults to 1.0.
            learning_rate (float, optional): Learning rate for the optimizer. Defaults to 5e-4.
            log_dir (str, optional): Directory for logging. Defaults to "logs".
        """
        # Call DQNTrainer's init but pass network=None as we create QRNetwork here.
        # The optimizer will also be re-initialized.
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
            network=None,  # Will be overridden
            log_dir=log_dir,
            # batch_size is a parameter of DQNTrainer's __init__, should pass it if not using default
            # DQNTrainer's default batch_size is 64. If we want to control it, add to QRDQN params.
        )

        self.num_quantiles = num_quantiles
        self.kappa = kappa

        # Initialize Q-network and Target Q-network
        self.q_network = QRNetwork(
            state_size, action_size, self.num_quantiles, network_hidden_sizes
        ).to(self.device)
        self.target_network = QRNetwork(
            state_size, action_size, self.num_quantiles, network_hidden_sizes
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network should be in eval mode

        # Re-initialize the optimizer with the QRNetwork parameters and the specified learning rate
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Precompute tau values for quantile regression loss
        # tau_i = (2i - 1) / (2N) for i = 1, ..., N (or (2i+1)/(2N) for i=0,...,N-1)
        # Using (2i+1)/(2N) for i = 0, ..., N-1
        i_tensor = torch.arange(
            0, self.num_quantiles, device=self.device, dtype=torch.float32
        )
        self.tau = ((2 * i_tensor + 1) / (2 * self.num_quantiles)).unsqueeze(
            0
        )  # Shape: (1, num_quantiles)

    def learn(self):
        """
        Performs a learning step for QR-DQN.
        Samples a batch from the replay buffer, calculates the quantile Huber loss,
        and updates the Q-network.

        Returns:
            float: The calculated loss value for this learning step.
        """
        # pdb.set_trace()
        # ipdb.set_trace()
        # pdbpp.set_trace()
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # Current Quantiles (Z(s,a))
        # current_q_quantiles shape: (batch_size, action_size, num_quantiles)
        current_q_quantiles = self.q_network(states)

        # Expand actions to gather the quantiles for the taken actions
        # actions shape: (batch_size)
        # actions_expanded shape: (batch_size, 1, num_quantiles)
        actions_expanded = actions[:, None, None].expand(-1, -1, self.num_quantiles)

        # current_sa_quantiles shape: (batch_size, num_quantiles)
        current_sa_quantiles = current_q_quantiles.gather(1, actions_expanded).squeeze(
            1
        )

        # Target Quantiles (Target Z'(s',a'))
        with torch.no_grad():
            # Get mean Q-values for next states from the target network for action selection (Double DQN style)
            # Using main network for action selection (argmax Q_online(s',a))
            # then use target network for value evaluation (Z_target(s', argmax Q_online(s',a)))

            # Option 1: Standard DQN target (argmax from target network)
            # next_mean_q_values = self.target_network.get_mean_q_values(next_states) # (B, A)
            # next_actions = torch.argmax(next_mean_q_values, dim=1, keepdim=True)    # (B, 1)

            # Option 2: Double DQN style target selection (argmax from online network, values from target)
            # This is generally preferred.
            next_mean_q_values_online = self.q_network.get_mean_q_values(
                next_states
            )  # (B, A)
            next_actions = torch.argmax(
                next_mean_q_values_online, dim=1, keepdim=True
            )  # (B, 1)

            # Get full quantile distributions for next states from the target network
            # next_q_quantiles shape: (batch_size, action_size, num_quantiles)
            next_q_quantiles_target = self.target_network(next_states)

            # Gather the quantiles corresponding to the selected next_actions
            # next_actions_expanded shape: (batch_size, 1, num_quantiles)
            next_actions_expanded = next_actions.unsqueeze(-1).expand(
                -1, -1, self.num_quantiles
            )

            # next_sa_quantiles_optimal shape: (batch_size, num_quantiles)
            next_sa_quantiles_optimal = next_q_quantiles_target.gather(
                1, next_actions_expanded
            ).squeeze(1)

            # Compute Bellman target for quantiles: T Z(s,a) = r + gamma * Z(s', a_next_optimal)
            # rewards shape: (batch_size, 1), dones shape: (batch_size, 1)
            # target_sa_quantiles shape: (batch_size, num_quantiles)
            target_sa_quantiles = rewards[
                :, None
            ] + self.gamma * next_sa_quantiles_optimal * (1 - dones.float()[:, None])
            # No detach needed here as it's already within torch.no_grad()

        # Loss Calculation
        # td_error (u_theta_ij) shape: (batch_size, num_quantiles_target, num_quantiles_current)
        # target_sa_quantiles.unsqueeze(2) -> (B, N, 1)
        # current_sa_quantiles.unsqueeze(1) -> (B, 1, N)
        # Resulting td_error shape: (B, N, N) where N is num_quantiles
        td_error = target_sa_quantiles.unsqueeze(2) - current_sa_quantiles.unsqueeze(1)

        # Huber loss calculation
        # huber_loss_case shape: (B, N, N)
        huber_loss_case = (torch.abs(td_error) <= self.kappa).float()
        quadratic_term = 0.5 * td_error.pow(2)
        linear_term = self.kappa * (torch.abs(td_error) - 0.5 * self.kappa)

        # element_wise_huber_loss (L_kappa(u_theta_ij)) shape: (B, N, N)
        element_wise_huber_loss = (
            huber_loss_case * quadratic_term + (1 - huber_loss_case) * linear_term
        )

        # Quantile regression loss
        # self.tau shape: (1, N)
        # self.tau.unsqueeze(-1) shape: (1, N, 1) (for broadcasting with td_error's target quantiles dimension)
        # (td_error < 0).float() shape: (B, N, N)
        # abs_tau_minus_indicator shape: (B, N, N)
        abs_tau_minus_indicator = torch.abs(
            self.tau.unsqueeze(-1) - (td_error < 0).float()
        )

        # quantile_huber_loss shape: (B, N, N)
        quantile_huber_loss = abs_tau_minus_indicator * element_wise_huber_loss

        # Sum over current quantiles (dim 2), mean over target quantiles (dim 1), then mean over batch.
        # The original paper sums over j (dim=2, current_sa_quantiles) and averages over i (dim=1, target_sa_quantiles).
        # Loss = sum_i sum_j L_ij / N (average over i)
        # Or average over batch, sum over i, sum over j.
        # The common implementation is: mean over batch, sum over target quantiles (dim1), mean over current quantiles (dim2)
        # Let's follow a common implementation: mean over batch, mean over target quantiles (dim 1), sum over current quantiles (dim 2)
        # Or, more simply: sum over current quantiles (dim 2), then mean over everything else.
        # loss = quantile_huber_loss.sum(dim=2).mean()
        # The paper (https://arxiv.org/pdf/1710.10044.pdf) suggests sum_j rho_tau_i(u_ij)
        # and then average this over i and the batch.
        # So, sum over dim 2 (current_sa_quantiles), then mean over dim 1 (target_sa_quantiles / self.tau), then mean over batch.
        # loss = (
        #     quantile_huber_loss.mean(dim=1).sum(dim=1).mean()
        # )  # mean(dim=1) is over target quantiles, sum(dim=1) is over current quantiles
        loss = quantile_huber_loss.sum(dim=2).mean()  # sum over current quantiles, mean over batch

        # Optimizer Step
        self.optimizer.zero_grad()
        loss.backward()
        # Optional: Gradient clipping
        # Optional: torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()

        return loss.item()

    def select_action(self, state: np.ndarray) -> int:
        """
        Selects an action using an epsilon-greedy policy based on mean Q-values.
        This method overrides the `select_action` method in DQNTrainer to use
        the mean of the quantile distribution for action selection, which is
        appropriate for QR-DQN.

        Args:
            state (np.ndarray): The current state of the environment.
                                Shape: (state_size,)
        Returns:
            int: The selected action (an integer from 0 to action_size-1).
        """
        # ipdb.set_trace()
        if np.random.rand() < self.epsilon:
            # Explore: select a random action
            return np.random.randint(self.action_size)
        else:
            # Exploit: select the action with the highest mean Q-value
            # Convert state to a PyTorch tensor, add batch dimension, and move to device
            # Ensure state is float, as expected by most PyTorch models.
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

            # Set the network to evaluation mode for deterministic output during inference
            self.q_network.eval()
            with torch.no_grad():
                # Get mean Q-values from the QR-DQN network
                mean_q_values = self.q_network.get_mean_q_values(
                    state_tensor
                )  # Shape: (1, action_size)
            # Set the network back to training mode
            self.q_network.train()

            # Select the action with the highest mean Q-value
            # .argmax(dim=1) returns indices for each item in batch; .item() gets Python number
            action = mean_q_values.argmax(dim=1).item()
            return action

    # Note: The `train_online` method is inherited from DQNTrainer.
    # By overriding `select_action` here, `train_online` will now use this
    # QR-DQN specific action selection mechanism when it calls `self.select_action(state, self.epsilon)`.
