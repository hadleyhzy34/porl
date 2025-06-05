import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import ipdb


class IQNNetwork(nn.Module):
    def __init__(self, state_size, action_size, embedding_dim=64, hidden_size=512):
        super(IQNNetwork, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.embedding_dim = embedding_dim

        # Feature extraction network
        self.feature_net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # Quantile embedding network
        self.quantile_embedding = nn.Linear(embedding_dim, hidden_size)

        # Final value network
        self.value_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, states, quantiles):
        """[TODO:description]

        Args:
            states ([TODO:parameter]): [TODO:description]
            quantiles ([TODO:parameter]): [TODO:description]

        Returns:
            [TODO:return]
        """
        # ipdb.set_trace()
        batch_size = states.size(0)
        num_quantiles = quantiles.size(1)

        # Extract state features
        state_features = self.feature_net(states)  # [batch_size, hidden_size]

        # Create quantile embeddings using cosine embedding
        quantile_embed = self.get_quantile_embedding(
            quantiles
        )  # [batch_size, num_quantiles, embedding_dim]
        quantile_embed = self.quantile_embedding(
            quantile_embed
        )  # [batch_size, num_quantiles, hidden_size]

        # Combine state features with quantile embeddings
        state_features = state_features.unsqueeze(1).expand(
            -1, num_quantiles, -1
        )  # [batch_size, num_quantiles, hidden_size]
        combined = state_features * quantile_embed  # Element-wise multiplication

        # Get quantile values for each action
        quantile_values = self.value_net(
            combined
        )  # [batch_size, num_quantiles, action_size]

        return quantile_values

    def get_quantile_embedding(self, quantiles):
        """Create cosine embeddings for quantiles"""
        # ipdb.set_trace()
        batch_size, num_quantiles = quantiles.shape

        # Create embedding indices
        embedding_indices = torch.arange(
            1, self.embedding_dim + 1, dtype=torch.float32, device=quantiles.device
        )
        embedding_indices = embedding_indices.view(1, 1, -1)  # [1, 1, embedding_dim]

        # Expand quantiles for embedding
        quantiles = quantiles.unsqueeze(-1)  # [batch_size, num_quantiles, 1]

        # Compute cosine embeddings
        embeddings = torch.cos(
            np.pi * embedding_indices * quantiles
        )  # [batch_size, num_quantiles, embedding_dim]

        return embeddings


# class IQNNetwork(nn.Module):
#     def __init__(self, state_dim, action_dim, embedding_dim=64, num_quantiles=8, dueling=True):
#         super(IQNNetwork, self).__init__()
#         self.state_dim = state_dim
#         self.action_dim = action_dim
#         self.embedding_dim = embedding_dim
#         self.num_quantiles = num_quantiles
#         self.dueling = dueling
#
#         # Cosine embedding layer
#         self.cosine_embedding = nn.Linear(embedding_dim, 256) # Output size matches feature_layer input
#
#         # Feature extraction layers
#         self.feature_layer = nn.Sequential(
#             nn.Linear(state_dim, 256),
#             nn.ReLU(),
#         )
#
#         if self.dueling:
#             # Advantage stream
#             self.advantage_fc = nn.Linear(256, action_dim * num_quantiles)
#             # Value stream
#             self.value_fc = nn.Linear(256, num_quantiles)
#         else:
#             self.fc_q = nn.Linear(256, action_dim * num_quantiles)
#
#     def forward(self, state, taus):
#         # state: (batch_size, state_dim)
#         # taus: (batch_size, num_quantiles) or (batch_size, N, num_quantiles) for per-quantile processing
#
#         # State feature extraction
#         state_features = self.feature_layer(state) # (batch_size, 256)
#
#         # Cosine embedding for taus
#         # taus shape: (batch_size, num_quantiles_tau, 1) where num_quantiles_tau is the number of tau samples
#         # We want to process each tau, so we might need to adjust dimensions
#         # Let's assume taus is (batch_size, num_quantiles_tau)
#         # We need to generate embeddings for each tau
#
#         batch_size = state.shape[0]
#         # If taus is (batch_size, num_quantiles), reshape for broadcasting later
#         # This might need adjustment based on how taus are sampled and passed
#         if len(taus.shape) == 2: # (batch_size, num_tau_samples)
#             num_tau_samples = taus.shape[1]
#             taus = taus.unsqueeze(-1) # -> (batch_size, num_tau_samples, 1)
#         elif len(taus.shape) == 3: # (batch_size, N, num_tau_samples) -> already (batch_size, N, K)
#             num_tau_samples = taus.shape[2]
#         else:
#             raise ValueError("taus shape not supported")
#
#
#         # Prepare pi * i for cosine calculation
#         #pis = torch.arange(1, self.embedding_dim + 1, device=taus.device).float() * math.pi
#         #pis = pis.view(1, 1, 1, self.embedding_dim) # (1, 1, 1, embedding_dim)
#         #taus_expanded = taus.unsqueeze(-1) # (batch_size, num_tau_samples, 1, 1) or (batch_size, N, num_tau_samples, 1)
#
#         # Cosine features: cos(pi * i * tau)
#         #cosine_basis = torch.cos(taus_expanded * pis) # (batch_size, num_tau_samples, 1, embedding_dim) or (batch_size, N, num_tau_samples, embedding_dim)
#
#         # Simpler cosine embedding:
#         # Assuming taus is (batch_size, num_tau_samples, 1)
#         # We want to get (batch_size, num_tau_samples, embedding_dim)
#
#         # Correct cosine embedding generation
#         # taus shape: (batch_size, num_tau_samples, 1)
#         # pis shape: (1, embedding_dim)
#         pis = torch.arange(0, self.embedding_dim, device=taus.device, dtype=torch.float32) * math.pi
#         pis = pis.view(1, 1, self.embedding_dim) # (1, 1, embedding_dim)
#
#         # taus: (batch_size, num_tau_samples, 1)
#         # cosine_embedding_input: (batch_size, num_tau_samples, embedding_dim)
#         cosine_embedding_input = torch.cos(taus * pis)
#
#
#         # Apply linear layer to cosine features
#         # Input to self.cosine_embedding: (batch_size * num_tau_samples, embedding_dim)
#         # Output: (batch_size * num_tau_samples, 256)
#         if len(cosine_embedding_input.shape) == 3: # (batch_size, num_tau_samples, embedding_dim)
#             tau_embeddings = F.relu(self.cosine_embedding(cosine_embedding_input.view(-1, self.embedding_dim)))
#             tau_embeddings = tau_embeddings.view(batch_size, num_tau_samples, -1) # (batch_size, num_tau_samples, 256)
#         else: # (batch_size, N, num_tau_samples, embedding_dim)
#             original_shape = cosine_embedding_input.shape
#             tau_embeddings = F.relu(self.cosine_embedding(cosine_embedding_input.view(-1, self.embedding_dim)))
#             tau_embeddings = tau_embeddings.view(original_shape[0], original_shape[1], original_shape[2], -1) # (batch_size, N, num_tau_samples, 256)
#
#
#         # Element-wise multiplication with state features
#         # state_features: (batch_size, 256)
#         # tau_embeddings: (batch_size, num_tau_samples, 256)
#         # We need to expand state_features to match tau_embeddings dimensions for multiplication
#         state_features_expanded = state_features.unsqueeze(1) # (batch_size, 1, 256)
#
#         # combined_features: (batch_size, num_tau_samples, 256)
#         combined_features = state_features_expanded * tau_embeddings
#
#         if self.dueling:
#             # Advantage stream
#             # Input: (batch_size * num_tau_samples, 256)
#             # Output: (batch_size * num_tau_samples, action_dim * num_quantiles_out)
#             # where num_quantiles_out is the number of quantiles predicted by the network (self.num_quantiles)
#             # The self.num_quantiles here is actually the K in IQN paper, the number of output quantiles
#             # The num_tau_samples is N in IQN paper, the number of sampled taus for input
#
#             # Reshape combined_features for linear layer
#             # (batch_size * num_tau_samples, 256)
#             advantage_flat = self.advantage_fc(combined_features.view(-1, 256))
#             # Reshape to (batch_size, num_tau_samples, action_dim, self.num_quantiles)
#             advantage = advantage_flat.view(batch_size, num_tau_samples, self.action_dim, self.num_quantiles)
#
#             # Value stream
#             # Input: (batch_size * num_tau_samples, 256)
#             # Output: (batch_size * num_tau_samples, self.num_quantiles)
#             value_flat = self.value_fc(combined_features.view(-1, 256))
#             # Reshape to (batch_size, num_tau_samples, 1, self.num_quantiles)
#             value = value_flat.view(batch_size, num_tau_samples, 1, self.num_quantiles)
#
#             # Combine advantage and value
#             # Q_values = V + (A - mean(A))
#             # mean_advantage shape: (batch_size, num_tau_samples, 1, self.num_quantiles)
#             mean_advantage = advantage.mean(dim=2, keepdim=True)
#             quantiles = value + (advantage - mean_advantage) # (batch_size, num_tau_samples, action_dim, self.num_quantiles)
#         else:
#             # Input: (batch_size * num_tau_samples, 256)
#             # Output: (batch_size * num_tau_samples, action_dim * self.num_quantiles)
#             quantiles_flat = self.fc_q(combined_features.view(-1, 256))
#             # Reshape to (batch_size, num_tau_samples, action_dim, self.num_quantiles)
#             quantiles = quantiles_flat.view(batch_size, num_tau_samples, self.action_dim, self.num_quantiles)
#
#         # quantiles: (batch_size, num_tau_samples, action_dim, self.num_quantiles)
#         # We typically want to take the mean over the sampled input taus (num_tau_samples) for action selection
#         # Or use all of them for the loss calculation.
#         # For action selection, Q(s,a) = E[Z_tau(s,a)] which is approximated by mean over Z_tau_samples(s,a)
#         # The output of this network is Z_tau(s,a) for each of the `self.num_quantiles` output quantiles,
#         # for each of the `num_tau_samples` input taus.
#
#         return quantiles
#
#     def get_q_values(self, state, taus):
#         # Returns Q(s,a) by taking the mean over the output quantiles for each input tau sample
#         # quantiles from forward: (batch_size, num_tau_samples, action_dim, self.num_quantiles)
#         quantiles = self.forward(state, taus)
#         # q_values: (batch_size, num_tau_samples, action_dim)
#         q_values = quantiles.mean(dim=3)
#         return q_values
#
if __name__ == "__main__":
    # Example Usage
    batch_size = 4
    state_dim = 8  # Example state dimension for LunarLander
    action_dim = 4  # Example action dimension for LunarLander
    embedding_dim = 64
    num_quantiles_output = 8  # K in IQN paper (number of quantiles network outputs)
    num_tau_samples = 32  # N in IQN paper (number of taus sampled for input)

    # Create network
    iqn_net = IQNNetwork(
        state_dim, action_dim, embedding_dim, num_quantiles_output, dueling=True
    )

    # Create dummy state and taus
    dummy_state = torch.randn(batch_size, state_dim)
    # Sample taus uniformly, e.g. (batch_size, num_tau_samples)
    dummy_taus = torch.rand(batch_size, num_tau_samples)

    # Forward pass
    # output_quantiles shape: (batch_size, num_tau_samples, action_dim, num_quantiles_output)
    output_quantiles = iqn_net(dummy_state, dummy_taus)
    print("Output Quantiles Shape:", output_quantiles.shape)

    # Get Q-values (mean over output quantiles for each input tau)
    # q_values shape: (batch_size, num_tau_samples, action_dim)
    q_values_per_tau_sample = iqn_net.get_q_values(dummy_state, dummy_taus)
    print("Q-values per tau sample Shape:", q_values_per_tau_sample.shape)

    # For action selection, typically mean over the num_tau_samples as well
    # final_q_values shape: (batch_size, action_dim)
    final_q_values = q_values_per_tau_sample.mean(dim=1)
    print("Final Q-values for action selection Shape:", final_q_values.shape)
    selected_actions = torch.argmax(final_q_values, dim=1)
    print("Selected Actions:", selected_actions)

    # Test with dueling=False
    iqn_net_no_dueling = IQNNetwork(
        state_dim, action_dim, embedding_dim, num_quantiles_output, dueling=False
    )
    output_quantiles_no_dueling = iqn_net_no_dueling(dummy_state, dummy_taus)
    print("Output Quantiles Shape (No Dueling):", output_quantiles_no_dueling.shape)
    q_values_no_dueling = iqn_net_no_dueling.get_q_values(dummy_state, dummy_taus)
    print("Q-values per tau sample Shape (No Dueling):", q_values_no_dueling.shape)
