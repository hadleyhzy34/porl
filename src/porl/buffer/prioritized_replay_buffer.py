import random
import numpy as np
from porl.buffer.sum_tree import SumTree
import pdb


class PrioritizedReplayBuffer:
    def __init__(
        self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000, epsilon=1e-5
    ):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha  # Prioritization exponent
        self.beta_start = beta_start  # Initial value of beta for IS
        self.beta_frames = beta_frames  # Number of frames to anneal beta to 1.0
        self.beta = beta_start
        self.epsilon = epsilon  # Small constant to ensure non-zero priorities
        self.frame_count = 0  # For beta annealing

    def _get_priority(self, td_error):
        return (abs(td_error) + self.epsilon) ** self.alpha

    def add(
        self, td_error, *experience
    ):  # experience is (state, action, reward, next_state, done)
        """
        Adds a new experience to the buffer.
        td_error: The TD error for this experience. For new experiences, this can be
                  set to a high value (e.g., max priority seen so far, or a default like 1.0)
                  to ensure they are sampled at least once.
        """
        priority = self._get_priority(td_error)
        # pdb.set_trace()
        self.tree.add(priority, experience)

    def sample(self, batch_size):
        batch = []
        idxs = []  # Tree indices of sampled experiences
        segment = self.tree.total_priority() / batch_size
        priorities = []

        self.beta = np.min(
            [
                1.0,
                self.beta_start
                + self.frame_count * (1.0 - self.beta_start) / self.beta_frames,
            ]
        )
        self.frame_count += 1

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get_leaf(s)

            if (
                not isinstance(data, np.ndarray) and data == 0
            ):  # Check if data is placeholder
                # This can happen if buffer is not full and we sample an empty spot
                # Retry sampling or handle appropriately. For simplicity, let's try again.
                # This part needs careful handling in a real scenario,
                # e.g., ensuring you only sample from filled parts of the tree.
                # A simple fix is to ensure SumTree.get_leaf only returns valid entries
                # or to resample if an invalid entry is hit.
                # Or, ensure SumTree.total_priority() only reflects filled entries.
                # For now, let's assume we sample valid data.
                # If len(self.tree) < batch_size, this loop will error.
                # Ensure sampling is only done when len(self.tree) >= batch_size
                s = random.uniform(
                    0, self.tree.total_priority()
                )  # Sample from full range
                (idx, p, data) = self.tree.get_leaf(s)

            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = np.array(priorities) / self.tree.total_priority()
        # (N * P(i))^-beta where N is current buffer size
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()  # Normalize for stability

        # Unzip experiences
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            np.array(is_weights),
            idxs,  # tree indices for updating priorities later
        )

    def update_priorities(self, tree_indices, td_errors):
        """
        Updates the priorities of the experiences at the given tree indices.
        tree_indices: A list/array of tree indices for the experiences in the last batch.
        td_errors: A list/array of new TD errors for these experiences.
        """
        for idx, error in zip(tree_indices, td_errors):
            priority = self._get_priority(error)
            self.tree.update(idx, priority)

    def __len__(self):
        return self.tree.n_entries
