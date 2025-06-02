import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# Assuming IQNPolicy is in src.porl.policy.iqn
# from src.porl.policy.iqn import IQNPolicy
# Assuming ReplayBuffer is available, e.g., from src.porl.buffer.replaybuffer
# from src.porl.buffer.replaybuffer import ReplayBuffer

class IQNTrainer:
    def __init__(self,
                 policy, # IQNPolicy instance
                 learning_rate=5e-4,
                 gamma=0.99,
                 batch_size=32,
                 buffer_size=100000,
                 target_update_frequency=100, # Steps
                 kappa=1.0, # For Huber loss in quantile regression
                 num_quantiles_loss_samples_N_prime=32, # N' in IQN paper, for current quantiles
                 num_quantiles_loss_samples_N_double_prime=32, # N'' in IQN paper, for target quantiles
                 device='cpu'):

        self.policy = policy
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        self.kappa = kappa
        self.num_quantiles_loss_samples_N_prime = num_quantiles_loss_samples_N_prime
        self.num_quantiles_loss_samples_N_double_prime = num_quantiles_loss_samples_N_double_prime
        self.device = device

        self.optimizer = optim.Adam(self.policy.network.parameters(), lr=learning_rate)
        # self.replay_buffer = ReplayBuffer(buffer_size, batch_size, device=device) # Example buffer

        self.train_step_counter = 0

    def quantile_huber_loss(self, td_errors, taus):
        # td_errors: (batch_size, num_tau_samples, num_target_tau_samples, num_quantiles_network_output)
        # taus: (batch_size, num_tau_samples, 1, 1)
        # Note: In IQN paper, taus are (batch_size, N', 1)
        # td_errors (u) shape: (batch_size, N', N'', K)
        # taus shape: (batch_size, N', 1, 1)

        # Pairwise quantile regression loss
        # td_errors is (theta_target - theta_current)
        # huber_loss part
        abs_td_errors = torch.abs(td_errors)
        huber_loss = torch.where(abs_td_errors <= self.kappa,
                                 0.5 * td_errors.pow(2),
                                 self.kappa * (abs_td_errors - 0.5 * self.kappa))
        # huber_loss shape: (batch_size, N', N'', K)

        # Quantile regression part
        # taus are for the current quantiles (N')
        # Indicator function I(u < 0)
        indicator = (td_errors < 0).float() # (batch_size, N', N'', K)
        # delta_taus = taus - I(u < 0)
        # taus shape: (batch_size, N', 1, 1)
        # indicator shape: (batch_size, N', N'', K)
        # td_errors_sign_term = (taus - indicator) * td_errors # This is wrong, should be (taus - I(u<0))u
        # Corrected:
        # (batch_size, N', 1, 1) - (batch_size, N', N'', K) -> (batch_size, N', N'', K)
        quantile_regression_factor = torch.abs(taus - indicator) # |tau - I(u < 0)|

        # element_wise_loss = quantile_regression_factor * huber_loss / self.kappa # Eq (4) in paper (division by kappa is part of QR-DQN, not always in IQN)
        element_wise_loss = quantile_regression_factor * huber_loss # More common form for IQN based on QR-DQN loss structure

        # Sum over K output quantiles from the network
        loss = element_wise_loss.sum(dim=3) # (batch_size, N', N'')

        # Mean over N'' (target tau samples) and N' (current tau samples)
        loss = loss.mean(dim=2).mean(dim=1) # (batch_size,)

        return loss.mean() # Mean over batch

    def train_step(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        # states, actions, rewards, next_states, dones = self.replay_buffer.sample()

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device) # (batch_size, 1)
        rewards = torch.FloatTensor(rewards).to(self.device) # (batch_size, 1)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device) # (batch_size, 1)

        # Sample taus for current value estimation (N')
        # taus_prime: (batch_size, N', 1)
        taus_prime = torch.rand(states.shape[0], self.num_quantiles_loss_samples_N_prime, 1).to(self.device)

        # Get current quantiles Z_theta(s,a; tau')
        # current_quantiles_all_actions: (batch_size, N', action_dim, K)
        current_quantiles_all_actions = self.policy.get_network_quantiles(states, taus_prime.squeeze(-1))

        # Gather quantiles for actions taken
        # actions need to be shaped for gather: (batch_size, N', 1, K)
        actions_expanded = actions.unsqueeze(1).unsqueeze(-1).expand(-1, self.num_quantiles_loss_samples_N_prime, -1, current_quantiles_all_actions.shape[-1])
        # current_quantiles: (batch_size, N', K)
        current_quantiles = current_quantiles_all_actions.gather(2, actions_expanded).squeeze(2)


        with torch.no_grad():
            # Sample taus for target value estimation (N'')
            # taus_double_prime: (batch_size, N'', 1)
            taus_double_prime = torch.rand(states.shape[0], self.num_quantiles_loss_samples_N_double_prime, 1).to(self.device)

            # Get next state Q-values from target network using N'' tau samples for policy evaluation
            # next_q_values_per_tau: (batch_size, N'', action_dim)
            next_q_values_per_tau = self.policy.target_network.get_q_values(next_states, taus_double_prime.squeeze(-1))
            # next_q_values: (batch_size, action_dim) - mean over N'' samples
            next_q_values = next_q_values_per_tau.mean(dim=1)

            # Select best actions in next state based on mean Q-values
            next_actions = torch.argmax(next_q_values, dim=1, keepdim=True) # (batch_size, 1)

            # Get target network quantiles Z_theta_target(s',a*; tau'') for the selected next actions
            # target_quantiles_all_actions: (batch_size, N'', action_dim, K)
            target_quantiles_all_actions = self.policy.get_target_network_quantiles(next_states, taus_double_prime.squeeze(-1))

            # Gather quantiles for next_actions
            # next_actions_expanded: (batch_size, N'', 1, K)
            next_actions_expanded = next_actions.unsqueeze(1).unsqueeze(-1).expand(-1, self.num_quantiles_loss_samples_N_double_prime, -1, target_quantiles_all_actions.shape[-1])
            # target_quantiles: (batch_size, N'', K)
            target_quantiles = target_quantiles_all_actions.gather(2, next_actions_expanded).squeeze(2)

            # Calculate target quantile values: r + gamma * Z_theta_target(s',a*; tau'')
            # rewards: (batch_size, 1) -> (batch_size, 1, 1)
            # dones: (batch_size, 1) -> (batch_size, 1, 1)
            # target_quantiles: (batch_size, N'', K)
            # td_target_quantiles: (batch_size, N'', K)
            td_target_quantiles = rewards.unsqueeze(-1) + self.gamma * target_quantiles * (1 - dones.unsqueeze(-1))


        # Calculate TD errors (Quantile differences)
        # current_quantiles: (batch_size, N', K) -> (batch_size, N', 1, K)
        # td_target_quantiles: (batch_size, N'', K) -> (batch_size, 1, N'', K)
        # td_errors: (batch_size, N', N'', K)
        td_errors = td_target_quantiles.unsqueeze(1) - current_quantiles.unsqueeze(2)

        # Calculate quantile Huber loss
        # taus_prime for loss: (batch_size, N', 1, 1)
        loss = self.quantile_huber_loss(td_errors, taus_prime.unsqueeze(-1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.policy.network.parameters(), max_norm=10.0) # Optional gradient clipping
        self.optimizer.step()

        self.train_step_counter += 1
        if self.train_step_counter % self.target_update_frequency == 0:
            self.policy.update_target_network()

        self.policy.update_epsilon() # Decay epsilon

        return loss.item()

if __name__ == '__main__':
    # Example Usage (requires IQNPolicy and a simple ReplayBuffer)

    # --- Mock IQNPolicy and ReplayBuffer for demonstration ---
    from src.porl.net.iqn_network import IQNNetwork # Make sure this path is correct

    class MockIQNPolicy(nn.Module): # Simplified for trainer test
        def __init__(self, state_dim, action_dim, device, num_quantiles_network_output=8, num_quantiles_policy_sample=32):
            super().__init__()
            self.network = IQNNetwork(state_dim, action_dim, num_quantiles=num_quantiles_network_output).to(device)
            self.target_network = IQNNetwork(state_dim, action_dim, num_quantiles=num_quantiles_network_output).to(device)
            self.target_network.load_state_dict(self.network.state_dict())
            self.device = device
            self.num_quantiles_policy_sample = num_quantiles_policy_sample

        def get_network_quantiles(self, states, taus_samples):
            return self.network(states, taus_samples)

        def get_target_network_quantiles(self, next_states, taus_samples):
            return self.target_network(next_states, taus_samples)

        def update_target_network(self):
            self.target_network.load_state_dict(self.network.state_dict())

        def update_epsilon(self): pass # Placeholder


    class MockReplayBuffer:
        def __init__(self, capacity, batch_size, state_dim, action_dim, device):
            self.device = device
            self.batch_size = batch_size
            self.pos = 0
            self.full = False
            self.capacity = capacity
            self.states = np.zeros((capacity, state_dim), dtype=np.float32)
            self.actions = np.zeros((capacity, 1), dtype=np.int64)
            self.rewards = np.zeros((capacity, 1), dtype=np.float32)
            self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
            self.dones = np.zeros((capacity, 1), dtype=np.float32)

        def push(self, s, a, r, ns, d):
            self.states[self.pos] = s
            self.actions[self.pos] = [a]
            self.rewards[self.pos] = [r]
            self.next_states[self.pos] = ns
            self.dones[self.pos] = [d]
            self.pos = (self.pos + 1) % self.capacity
            if self.pos == 0: self.full = True

        def sample(self):
            idxs = np.random.randint(0, self.capacity if self.full else self.pos, size=self.batch_size)
            return (
                torch.FloatTensor(self.states[idxs]).to(self.device),
                torch.LongTensor(self.actions[idxs]).to(self.device),
                torch.FloatTensor(self.rewards[idxs]).to(self.device),
                torch.FloatTensor(self.next_states[idxs]).to(self.device),
                torch.FloatTensor(self.dones[idxs]).to(self.device)
            )
        def __len__(self):
            return self.capacity if self.full else self.pos
    # --- End Mock ---

    state_dim = 8
    action_dim = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 16 # Smaller for quick test

    # Use the mock policy for the trainer
    mock_policy = MockIQNPolicy(state_dim, action_dim, device=device)

    trainer = IQNTrainer(
        policy=mock_policy,
        learning_rate=1e-4,
        batch_size=batch_size,
        device=device,
        num_quantiles_loss_samples_N_prime=8, # Smaller for quick test
        num_quantiles_loss_samples_N_double_prime=8 # Smaller for quick test
    )

    # Populate a mock buffer for a few samples
    mock_buffer = MockReplayBuffer(100, batch_size, state_dim, action_dim, device)
    for _ in range(batch_size * 2): # Fill enough for a few batches
        s = np.random.randn(state_dim)
        a = np.random.randint(action_dim)
        r = np.random.rand()
        ns = np.random.randn(state_dim)
        d = float(np.random.rand() > 0.9)
        mock_buffer.push(s, a, r, ns, d)

    if len(mock_buffer) >= batch_size:
        print("Starting dummy training step...")
        experiences = mock_buffer.sample()
        loss = trainer.train_step(experiences)
        print(f"Training step complete. Loss: {loss:.4f}")
        print(f"Policy epsilon: {trainer.policy.epsilon if hasattr(trainer.policy, 'epsilon') else 'N/A'}")
        print(f"Train step counter: {trainer.train_step_counter}")
    else:
        print("Buffer not full enough for a training step.")

    print("IQN Trainer example finished.")
