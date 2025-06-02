import torch
import torch.nn as nn
import numpy as np
from src.porl.net.iqn_network import IQNNetwork # Assuming the network is in this path

class IQNPolicy(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 embedding_dim=64,
                 num_quantiles_network_output=8, # K: Number of quantiles the network outputs
                 num_quantiles_policy_sample=32, # N: Number of taus sampled for policy evaluation (action selection)
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay=100000,
                 dueling_network=True,
                 device='cpu'):
        super(IQNPolicy, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_quantiles_policy_sample = num_quantiles_policy_sample
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.device = device

        self.network = IQNNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            embedding_dim=embedding_dim,
            num_quantiles=num_quantiles_network_output, # K
            dueling=dueling_network
        ).to(self.device)

        self.target_network = IQNNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            embedding_dim=embedding_dim,
            num_quantiles=num_quantiles_network_output, # K
            dueling=dueling_network
        ).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.eval()

        self.train_step_counter = 0

    def select_action(self, state, deterministic=False):
        if not deterministic and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            # For action selection, we sample N taus (num_quantiles_policy_sample)
            taus = torch.rand(1, self.num_quantiles_policy_sample).to(self.device) # (1, N)

            with torch.no_grad():
                # q_values_per_tau_sample: (1, N, action_dim)
                q_values_per_tau_sample = self.network.get_q_values(state_tensor, taus)
                # q_values: (1, action_dim) - take mean over N sampled taus
                q_values = q_values_per_tau_sample.mean(dim=1)
            return torch.argmax(q_values, dim=1).item()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon - (1.0 - self.epsilon_end) / self.epsilon_decay)
        # Alternative decay: self.epsilon * self.epsilon_decay_rate

    def update_target_network(self, tau=0.005):
        for target_param, local_param in zip(self.target_network.parameters(), self.network.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def get_network_quantiles(self, states, taus_samples):
        # states: (batch_size, state_dim)
        # taus_samples: (batch_size, num_tau_samples_for_loss)
        # Returns: (batch_size, num_tau_samples_for_loss, action_dim, num_quantiles_network_output)
        return self.network(states, taus_samples)

    def get_target_network_quantiles(self, next_states, taus_samples):
        # next_states: (batch_size, state_dim)
        # taus_samples: (batch_size, num_tau_samples_for_loss)
        # Returns: (batch_size, num_tau_samples_for_loss, action_dim, num_quantiles_network_output)
        return self.target_network(next_states, taus_samples)

if __name__ == '__main__':
    # Example Usage
    state_dim = 8
    action_dim = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = IQNPolicy(state_dim, action_dim, device=device, num_quantiles_network_output=8, num_quantiles_policy_sample=32)

    # Dummy state
    dummy_state_np = np.random.randn(state_dim)

    # Select action
    action = policy.select_action(dummy_state_np)
    print(f"Selected action: {action}")

    # Update epsilon (simulate some steps)
    for _ in range(1000):
        policy.update_epsilon()
    print(f"Epsilon after 1000 steps: {policy.epsilon:.4f}")

    # Update target network
    policy.update_target_network()
    print("Target network updated.")

    # Example of getting quantiles for loss calculation
    batch_size = 32
    num_tau_samples_for_loss = 64 # N' in IQN paper for loss

    dummy_states_batch = torch.randn(batch_size, state_dim).to(device)
    # For loss, taus are typically sampled per observation, so (batch_size, num_tau_samples_for_loss)
    dummy_taus_batch = torch.rand(batch_size, num_tau_samples_for_loss).to(device)

    # Get quantiles from current network
    # quantiles shape: (batch_size, num_tau_samples_for_loss, action_dim, num_quantiles_network_output)
    network_quantiles = policy.get_network_quantiles(dummy_states_batch, dummy_taus_batch)
    print("Network Quantiles Shape:", network_quantiles.shape)

    # Get quantiles from target network
    target_network_quantiles = policy.get_target_network_quantiles(dummy_states_batch, dummy_taus_batch)
    print("Target Network Quantiles Shape:", target_network_quantiles.shape)
