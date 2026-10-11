"""Configuration and Hyperparameters for Lunar Lander RL Agents."""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class EnvConfig:
    env_id: str = "LunarLander-v3"
    state_dim: int = 8
    action_dim: int = 4
    solved_score: float = 200.0
    max_steps_per_episode: int = 1000
    seed: int = 42


@dataclass
class BaselineDQNConfig:
    """Model 1: Vanilla DQN (Classic / Weak Baseline)."""
    model_name: str = "baseline_dqn"
    hidden_dims: Tuple[int, ...] = (64, 64)
    lr: float = 5e-4
    gamma: float = 0.99
    buffer_size: int = 100_000
    batch_size: int = 64
    target_update_freq: int = 1000  # Hard periodic copy
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay: float = 0.992
    learning_starts: int = 500
    train_freq: int = 4  # Infrequent training (less sample efficient)
    gradient_clip: float = 10.0
    max_episodes: int = 400
    eval_freq: int = 20
    eval_episodes: int = 10


@dataclass
class AdvancedDDQNConfig:
    """Model 2: Dueling Double DQN with Prioritized Experience Replay."""
    model_name: str = "advanced_dueling_ddqn"
    hidden_dims: Tuple[int, ...] = (256, 256)
    lr: float = 5e-4
    gamma: float = 0.99
    buffer_size: int = 100_000
    batch_size: int = 64
    tau: float = 0.005  # Polyak continuous soft target update
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay: float = 0.985  # Faster exploration decay
    learning_starts: int = 500
    train_freq: int = 1  # High sample efficiency: train on every step
    gradient_clip: float = 5.0
    
    # Prioritized Experience Replay (PER)
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_frames: int = 40_000
    per_eps: float = 1e-5
    
    max_episodes: int = 300
    eval_freq: int = 15
    eval_episodes: int = 10
