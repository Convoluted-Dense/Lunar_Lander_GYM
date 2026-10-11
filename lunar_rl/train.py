"""Training Pipeline for Lunar Lander Reinforcement Learning Agents.

Supports:
- Model 1: Baseline Vanilla DQN
- Model 2: Advanced Dueling Double DQN with PER
- Automated evaluation, rolling metrics tracking, and best model checkpointing.
"""

import os
import time
import json
import argparse
import numpy as np
import torch
import gymnasium as gym
from collections import deque
from typing import Dict, Any, List

from lunar_rl.config import EnvConfig, BaselineDQNConfig, AdvancedDDQNConfig
from lunar_rl.models.baseline_dqn import BaselineDQNAgent
from lunar_rl.models.dueling_ddqn import AdvancedDuelingDDQNAgent


def evaluate_agent(env: gym.Env, agent, num_episodes: int = 10) -> Dict[str, float]:
    """Run deterministic evaluation episodes without exploration."""
    eval_rewards = []
    landed_count = 0

    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action = agent.select_action(state, evaluate=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            ep_reward += reward

        eval_rewards.append(ep_reward)
        if ep_reward >= 150:  # Threshold representing successful soft touchdown
            landed_count += 1

    return {
        "mean_reward": float(np.mean(eval_rewards)),
        "std_reward": float(np.std(eval_rewards)),
        "min_reward": float(np.min(eval_rewards)),
        "max_reward": float(np.max(eval_rewards)),
        "landing_rate": float(landed_count / num_episodes),
    }


def train_model(
    model_type: str = "advanced",
    max_episodes: int = 500,
    seed: int = 42,
    checkpoints_dir: str = "checkpoints",
    reports_dir: str = "reports",
) -> Dict[str, Any]:
    """Execute complete training run for a selected RL model."""
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    env_cfg = EnvConfig(seed=seed)
    env = gym.make(env_cfg.env_id)
    eval_env = gym.make(env_cfg.env_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"[START] Initializing Training Run: {model_type.upper()}")
    print(f"   Environment: {env_cfg.env_id} | Device: {device}")
    print(f"   Max Episodes: {max_episodes} | Target Score: {env_cfg.solved_score}")
    print(f"=======================================================\n")

    if model_type == "baseline":
        cfg = BaselineDQNConfig(max_episodes=max_episodes)
        agent = BaselineDQNAgent(env_cfg, cfg, device=device)
        checkpoint_path = os.path.join(checkpoints_dir, "baseline_best.pt")
        metrics_path = os.path.join(reports_dir, "metrics_baseline.json")
    elif model_type == "advanced":
        cfg = AdvancedDDQNConfig(max_episodes=max_episodes)
        agent = AdvancedDuelingDDQNAgent(env_cfg, cfg, device=device)
        checkpoint_path = os.path.join(checkpoints_dir, "advanced_best.pt")
        metrics_path = os.path.join(reports_dir, "metrics_advanced.json")
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Telemetry storage
    scores_history: List[float] = []
    avg_scores_history: List[float] = []
    loss_history: List[float] = []
    recent_scores = deque(maxlen=100)
    best_avg_score = -float("inf")
    solved_episode = None

    start_time = time.time()

    for episode in range(1, max_episodes + 1):
        state, _ = env.reset(seed=seed + episode)
        ep_reward = 0.0
        ep_losses = []
        done = False

        while not done:
            action = agent.select_action(state, evaluate=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            loss = agent.step(state, action, reward, next_state, done)
            if loss is not None:
                ep_losses.append(loss)

            state = next_state
            ep_reward += reward

        agent.decay_epsilon()
        recent_scores.append(ep_reward)
        scores_history.append(ep_reward)

        current_avg = float(np.mean(recent_scores))
        avg_scores_history.append(current_avg)
        mean_ep_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
        loss_history.append(mean_ep_loss)

        # Check for best model checkpointing
        if len(recent_scores) >= 20 and current_avg > best_avg_score:
            best_avg_score = current_avg
            agent.save(checkpoint_path)

        # Check solved criterion
        if current_avg >= env_cfg.solved_score and solved_episode is None and len(recent_scores) >= 100:
            solved_episode = episode
            print(f"\n[SOLVED] Environment solved in Episode {episode} with 100-ep Avg: {current_avg:.2f} pts!\n")

        # Logging output
        if episode % 10 == 0 or episode == 1:
            elapsed = time.time() - start_time
            print(
                f"Ep {episode:4d}/{max_episodes} | "
                f"Score: {ep_reward:7.1f} | "
                f"Avg(100): {current_avg:7.1f} | "
                f"BestAvg: {best_avg_score:7.1f} | "
                f"Eps: {agent.epsilon:.3f} | "
                f"Loss: {mean_ep_loss:.4f} | "
                f"Elapsed: {elapsed:.1f}s"
            )

        # Periodic full evaluation
        if episode % cfg.eval_freq == 0:
            eval_metrics = evaluate_agent(eval_env, agent, cfg.eval_episodes)
            print(
                f"   [EVAL @ Ep {episode:4d}] "
                f"Mean Return: {eval_metrics['mean_reward']:7.1f} +/- {eval_metrics['std_reward']:.1f} | "
                f"Landing Rate: {eval_metrics['landing_rate']*100:.0f}%"
            )

    env.close()
    eval_env.close()

    total_time = time.time() - start_time
    print(f"\n[DONE] Training completed in {total_time:.1f}s. Best 100-Ep Average: {best_avg_score:.2f} pts.")

    # Save metrics JSON
    metrics = {
        "model_type": model_type,
        "max_episodes": max_episodes,
        "total_training_time_sec": total_time,
        "best_avg_score": best_avg_score,
        "solved_episode": solved_episode,
        "scores": scores_history,
        "avg_scores": avg_scores_history,
        "losses": loss_history,
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[SAVED] Telemetry metrics saved to: {metrics_path}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Lunar Lander RL Models")
    parser.add_argument(
        "--model",
        type=str,
        choices=["baseline", "advanced", "both"],
        default="advanced",
        help="Which model architecture to train",
    )
    parser.add_argument("--episodes", type=int, default=350, help="Number of training episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if args.model in ["baseline", "both"]:
        train_model("baseline", max_episodes=args.episodes, seed=args.seed)
    if args.model in ["advanced", "both"]:
        train_model("advanced", max_episodes=args.episodes, seed=args.seed)
