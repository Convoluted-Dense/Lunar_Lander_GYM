"""Proximal Policy Optimization (PPO) with Generalized Advantage Estimation (GAE) for Lunar Lander."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import Dict, Any, List, Tuple

from lunar_rl.config import EnvConfig


class ActorCriticNetwork(nn.Module):
    """Joint Actor-Critic Network for PPO."""

    def __init__(self, state_dim: int = 8, action_dim: int = 4, hidden_dim: int = 128):
        super().__init__()
        
        # Policy Network (Actor)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Value Network (Critic)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.actor(state)
        value = self.critic(state)
        return logits, value

    def get_action_and_value(self, state: torch.Tensor, action: torch.Tensor = None):
        logits = self.actor(state)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(state)


class PPOAgent:
    """PPO Agent with Clipped Objective and GAE-Lambda."""

    def __init__(
        self,
        env_config: EnvConfig,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        device: torch.device = None,
    ):
        self.env_cfg = env_config
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.network = ActorCriticNetwork(self.env_cfg.state_dim, self.env_cfg.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.network.actor(state_t)
            if evaluate:
                return int(logits.argmax(dim=-1).item())
            dist = Categorical(logits=logits)
            return int(dist.sample().item())

    def save(self, filepath: str):
        torch.save({"network": self.network.state_dict(), "optimizer": self.optimizer.state_dict()}, filepath)

    def load(self, filepath: str):
        ckpt = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(ckpt["network"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
