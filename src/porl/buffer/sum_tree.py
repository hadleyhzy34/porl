import numpy as np


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity  # Number of leaf nodes (i.e., experiences)
        # Tree structure:
        # Parent node: (idx - 1) // 2
        # Left child: 2 * idx + 1
        # Right child: 2 * idx + 2
        self.tree = np.zeros(2 * capacity - 1)  # Internal nodes + leaf nodes
        self.data = np.zeros(capacity, dtype=object)  # Stores the actual experiences
        self.data_pointer = 0  # Points to the next available slot in self.data
        self.n_entries = 0  # Current number of entries in the tree

    def _propagate(self, idx, change):
        """Propagates a change in priority up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:  # If not root
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        """
        Finds the leaf node corresponding to a given sum `s`.
        idx: current node index in the tree array
        s: the sum value we are searching for
        """
        left_child_idx = 2 * idx + 1
        right_child_idx = left_child_idx + 1

        if left_child_idx >= len(self.tree):  # Reached a leaf node
            return idx

        if s <= self.tree[left_child_idx]:
            return self._retrieve(left_child_idx, s)
        else:
            return self._retrieve(right_child_idx, s - self.tree[left_child_idx])

    def total_priority(self):
        """Returns the total priority (sum of all priorities), which is stored at the root."""
        return self.tree[0]

    def add(self, priority, data):
        """Adds a new experience with its priority."""
        tree_idx = (
            self.data_pointer + self.capacity - 1
        )  # Index in the tree array for the leaf

        self.data[self.data_pointer] = data  # Store experience
        self.update(tree_idx, priority)  # Update tree with new priority

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # Ring buffer
            self.data_pointer = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, tree_idx, priority):
        """Updates the priority of an existing experience at tree_idx."""
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        if tree_idx != 0:  # Avoid propagating from root if capacity is 1
            self._propagate(tree_idx, change)

    def get_leaf(self, s):
        """
        Gets the leaf index, priority, and data for a given sum `s`.
        s: a value sampled from [0, total_priority)
        """
        idx = self._retrieve(0, s)  # Get tree index of the leaf
        data_idx = idx - self.capacity + 1  # Convert tree index to data array index
        return idx, self.tree[idx], self.data[data_idx]

    def __len__(self):
        return self.n_entries
