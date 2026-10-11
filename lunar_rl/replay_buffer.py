"""Experience Replay Buffers for Reinforcement Learning.

Includes:
1. UniformReplayBuffer (for Model 1 Baseline DQN)
2. PrioritizedReplayBuffer with SumTree (for Model 2 Advanced Dueling DDQN)
"""

import numpy as np
import torch
from typing import Tuple, Dict, Any


class UniformReplayBuffer:
    """Standard Circular Experience Replay Buffer with Uniform Random Sampling."""

    def __init__(self, capacity: int, state_dim: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = np.random.randint(0, self.size, size=batch_size)

        states = torch.tensor(self.states[indices], dtype=torch.float32, device=self.device)
        actions = torch.tensor(self.actions[indices], dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor(self.rewards[indices], dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(self.next_states[indices], dtype=torch.float32, device=self.device)
        dones = torch.tensor(self.dones[indices], dtype=torch.float32, device=self.device).unsqueeze(1)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return self.size


class SumTree:
    """Binary SumTree Data Structure for O(log N) Priority Updates and Proportional Sampling."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data_pointer = 0
        self.size = 0

    def add(self, priority: float):
        tree_idx = self.data_pointer + self.capacity - 1
        self.update(tree_idx, priority)

        self.data_pointer = (self.data_pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, tree_idx: int, priority: float):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v: float) -> Tuple[int, float, int]:
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1

            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break

            if v <= self.tree[left_child_idx]:
                parent_idx = left_child_idx
            else:
                v -= self.tree[left_child_idx]
                parent_idx = right_child_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], data_idx

    @property
    def total_priority(self) -> float:
        return float(self.tree[0])


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay (PER) with Importance Sampling Weighting."""

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        device: torch.device,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100_000,
        eps: float = 1e-5,
    ):
        self.capacity = capacity
        self.device = device
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.eps = eps
        self.frame = 1
        self.max_priority = 1.0

        self.tree = SumTree(capacity)
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

    def beta(self) -> float:
        """Anneal beta linearly from beta_start to 1.0."""
        return min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        ptr = self.tree.data_pointer
        self.states[ptr] = state
        self.actions[ptr] = action
        self.rewards[ptr] = reward
        self.next_states[ptr] = next_state
        self.dones[ptr] = float(done)

        # New experiences get maximal priority to ensure initial sampling
        priority = self.max_priority ** self.alpha
        self.tree.add(priority)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, torch.Tensor]:
        self.frame += 1
        b_idx = np.empty((batch_size,), dtype=np.int32)
        tree_idx = np.empty((batch_size,), dtype=np.int32)
        priorities = np.empty((batch_size,), dtype=np.float32)

        segment = self.tree.total_priority / batch_size
        beta = self.beta()

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            v = np.random.uniform(a, b)
            t_idx, p, d_idx = self.tree.get_leaf(v)
            b_idx[i] = d_idx
            tree_idx[i] = t_idx
            priorities[i] = p

        sampling_probabilities = priorities / (self.tree.total_priority + 1e-10)
        is_weights = np.power(self.tree.size * sampling_probabilities + 1e-10, -beta)
        is_weights /= (is_weights.max() + 1e-10)  # Normalize for stability

        states = torch.tensor(self.states[b_idx], dtype=torch.float32, device=self.device)
        actions = torch.tensor(self.actions[b_idx], dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor(self.rewards[b_idx], dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(self.next_states[b_idx], dtype=torch.float32, device=self.device)
        dones = torch.tensor(self.dones[b_idx], dtype=torch.float32, device=self.device).unsqueeze(1)
        weights = torch.tensor(is_weights, dtype=torch.float32, device=self.device).unsqueeze(1)

        return states, actions, rewards, next_states, dones, tree_idx, weights

    def update_priorities(self, tree_indices: np.ndarray, td_errors: np.ndarray):
        """Update transition priorities with new TD error values."""
        for t_idx, td in zip(tree_indices, td_errors):
            priority = (np.abs(td) + self.eps) ** self.alpha
            self.tree.update(t_idx, priority)
            self.max_priority = max(self.max_priority, float(np.abs(td) + self.eps))

    def __len__(self) -> int:
        return self.tree.size
