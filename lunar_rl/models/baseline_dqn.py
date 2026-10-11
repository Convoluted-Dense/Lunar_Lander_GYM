"""Model 1: Classic Vanilla Deep Q-Network (Vanilla DQN).

Represents the classic / baseline RL architecture (Mnih et al., 2013/2015).
Key traits:
- Standard 2-layer MLP Q-network
- Maximization over next-state Q-values (subject to overestimation bias)
- Uniform random replay sampling
- Periodic hard target network synchronization
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Optional

from lunar_rl.config import BaselineDQNConfig, EnvConfig
from lunar_rl.replay_buffer import UniformReplayBuffer


class VanillaQNetwork(nn.Module):
    """Standard Multi-Layer Perceptron Q-Network."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims=(64, 64)):
        super().__init__()
        layers = []
        in_dim = state_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class BaselineDQNAgent:
    """Agent implementing Vanilla Deep Q-Learning."""

    def __init__(
        self,
        env_config: EnvConfig,
        agent_config: BaselineDQNConfig,
        device: Optional[torch.device] = None,
    ):
        self.env_cfg = env_config
        self.cfg = agent_config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Online Policy Network & Target Network
        self.policy_net = VanillaQNetwork(
            self.env_cfg.state_dim, self.env_cfg.action_dim, self.cfg.hidden_dims
        ).to(self.device)
        self.target_net = VanillaQNetwork(
            self.env_cfg.state_dim, self.env_cfg.action_dim, self.cfg.hidden_dims
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer & Loss
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.cfg.lr)
        self.criterion = nn.MSELoss()

        # Experience Replay Buffer (Uniform)
        self.buffer = UniformReplayBuffer(
            self.cfg.buffer_size, self.env_cfg.state_dim, self.device
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
        """Store transition and conditionally train."""
        self.buffer.add(state, action, reward, next_state, done)
        self.total_steps += 1

        loss = None
        if len(self.buffer) >= self.cfg.learning_starts and self.total_steps % self.cfg.train_freq == 0:
            loss = self._learn()

        # Periodic hard target network update
        if self.total_steps % self.cfg.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss

    def _learn(self) -> float:
        """Perform one step of standard Q-learning gradient descent."""
        states, actions, rewards, next_states, dones = self.buffer.sample(self.cfg.batch_size)

        # Current Q-values: Q(s, a; θ)
        current_q = self.policy_net(states).gather(1, actions)

        # Target Q-values: r + γ * max_a' Q(s', a'; θ^-) [Maximization Bias present]
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            target_q = rewards + (self.cfg.gamma * (1.0 - dones) * max_next_q)

        loss = self.criterion(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        if self.cfg.gradient_clip > 0:
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.cfg.gradient_clip)
        self.optimizer.step()

        self.updates_count += 1
        return float(loss.item())

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
