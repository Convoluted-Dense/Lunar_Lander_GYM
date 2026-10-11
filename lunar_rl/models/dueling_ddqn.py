"""Model 2: Advanced Dueling Double Deep Q-Network with Prioritized Experience Replay (Dueling DDQN + PER).

Integrates the three major modern improvements to Deep Q-Learning:
1. Double Q-Learning (DDQN) - eliminates maximization overestimation bias.
2. Dueling Network Architecture - decomposes state value V(s) and action advantage A(s, a).
3. Prioritized Experience Replay (PER) - samples transitions proportional to TD error magnitude.
4. Polyak Soft Target Updates - smooth continuous target network evolution.
5. Huber (Smooth L1) Loss - robustness against outliers and crash penalties.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Optional

from lunar_rl.config import AdvancedDDQNConfig, EnvConfig
from lunar_rl.replay_buffer import PrioritizedReplayBuffer


class DuelingQNetwork(nn.Module):
    """Dueling Neural Network Architecture separating State Value V(s) and Advantage A(s, a)."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims=(256, 256)):
        super().__init__()
        h_dim = hidden_dims[0]

        # Shared feature extractor
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, h_dim),
            nn.LayerNorm(h_dim),
            nn.ReLU(),
        )

        # State Value Stream V(s) -> scalar
        self.value_stream = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 1),
        )

        # Action Advantage Stream A(s, a) -> vector of size action_dim
        self.advantage_stream = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.feature_layer(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # Q(s, a) = V(s) + (A(s, a) - mean(A(s, a')))
        q_values = values + (advantages - advantages.mean(dim=-1, keepdim=True))
        return q_values


class AdvancedDuelingDDQNAgent:
    """Advanced Deep RL Agent combining Dueling Networks, Double Q-Learning, and PER."""

    def __init__(
        self,
        env_config: EnvConfig,
        agent_config: AdvancedDDQNConfig,
        device: Optional[torch.device] = None,
    ):
        self.env_cfg = env_config
        self.cfg = agent_config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Online Policy Net & Target Net
        self.policy_net = DuelingQNetwork(
            self.env_cfg.state_dim, self.env_cfg.action_dim, self.cfg.hidden_dims
        ).to(self.device)
        self.target_net = DuelingQNetwork(
            self.env_cfg.state_dim, self.env_cfg.action_dim, self.cfg.hidden_dims
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer with weight decay for stability
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.cfg.lr, weight_decay=1e-4)
        self.criterion = nn.SmoothL1Loss(reduction='none')  # Elementwise for IS weighting

        # Prioritized Experience Replay Buffer
        self.buffer = PrioritizedReplayBuffer(
            capacity=self.cfg.buffer_size,
            state_dim=self.env_cfg.state_dim,
            device=self.device,
            alpha=self.cfg.per_alpha,
            beta_start=self.cfg.per_beta_start,
            beta_frames=self.cfg.per_beta_frames,
            eps=self.cfg.per_eps,
        )

        # Exploration & Tracking
        self.epsilon = self.cfg.eps_start
        self.total_steps = 0
        self.updates_count = 0

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """Epsilon-greedy action selection."""
        if not evaluate and np.random.rand() < self.epsilon:
            return int(np.random.randint(self.env_cfg.action_dim))

        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            return int(q_values.argmax(dim=1).item())

    def step(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> Optional[float]:
        """Store experience in PER buffer and conditionally train."""
        self.buffer.add(state, action, reward, next_state, done)
        self.total_steps += 1

        loss = None
        if len(self.buffer) >= self.cfg.learning_starts and self.total_steps % self.cfg.train_freq == 0:
            loss = self._learn()

        return loss

    def _learn(self) -> float:
        """Double Q-learning update step with PER Importance Sampling weights."""
        states, actions, rewards, next_states, dones, tree_indices, is_weights = self.buffer.sample(self.cfg.batch_size)

        # Current Q-values: Q_online(s, a; θ)
        current_q = self.policy_net(states).gather(1, actions)

        # Double DQN Target:
        # 1. Select greedy action using ONLINE network: a* = argmax_a' Q_online(s', a'; θ)
        # 2. Evaluate that action using TARGET network: Q_target(s', a*; θ^-)
        with torch.no_grad():
            best_next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            target_next_q = self.target_net(next_states).gather(1, best_next_actions)
            target_q = rewards + (self.cfg.gamma * (1.0 - dones) * target_next_q)

        # Compute TD error for updating SumTree priorities
        td_errors = (target_q - current_q).detach().squeeze().cpu().numpy()
        self.buffer.update_priorities(tree_indices, td_errors)

        # Weighted Huber Loss: (1/B) * Σ [ w_i * SmoothL1(Q - y) ]
        elementwise_loss = self.criterion(current_q, target_q)
        weighted_loss = (is_weights * elementwise_loss).mean()

        self.optimizer.zero_grad()
        weighted_loss.backward()
        if self.cfg.gradient_clip > 0:
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.cfg.gradient_clip)
        self.optimizer.step()

        # Polyak Soft Target Network Update: θ^- ← τ θ + (1 - τ) θ^-
        self._soft_update_target(self.cfg.tau)

        self.updates_count += 1
        return float(weighted_loss.item())

    def _soft_update_target(self, tau: float):
        """Gradually blend online parameters into target network."""
        for target_param, online_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)

    def decay_epsilon(self):
        """Exponential decay of epsilon parameter."""
        self.epsilon = max(self.cfg.eps_end, self.epsilon * self.cfg.eps_decay)

    def save(self, filepath: str):
        """Save model checkpoint."""
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "total_steps": self.total_steps,
            },
            filepath,
        )

    def load(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint.get("epsilon", self.cfg.eps_end)
        self.total_steps = checkpoint.get("total_steps", 0)
