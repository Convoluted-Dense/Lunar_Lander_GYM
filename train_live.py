"""Live Interactive Training Script for Lunar Lander in OpenAI Gym / Gymnasium Window.

Allows the user to visually watch the agent learn episode-by-episode on screen!

Usage:
  python train_live.py --model advanced --episodes 100
  python train_live.py --model baseline --episodes 100
  python train_live.py --render-every 1   # Render every episode
"""

import os
import time
import argparse
import numpy as np
import torch
import gymnasium as gym
from collections import deque

from lunar_rl.config import EnvConfig, BaselineDQNConfig, AdvancedDDQNConfig
from lunar_rl.models.baseline_dqn import BaselineDQNAgent
from lunar_rl.models.dueling_ddqn import AdvancedDuelingDDQNAgent


def train_live_visual(
    model_type: str = "advanced",
    max_episodes: int = 200,
    render_every: int = 1,
    seed: int = 42,
):
    """Train RL model live with Gymnasium desktop rendering window."""
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    env_cfg = EnvConfig(seed=seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n=======================================================")
    print(f"[LIVE TRAINING] Launching OpenAI Gym / Gymnasium Window")
    print(f"  Model Architecture: {model_type.upper()}")
    print(f"  Max Episodes: {max_episodes}")
    print(f"  Render Frequency: Every {render_every} episode(s)")
    print(f"  Device: {device}")
    print("=======================================================\n")

    if model_type == "baseline":
        cfg = BaselineDQNConfig(max_episodes=max_episodes)
        agent = BaselineDQNAgent(env_cfg, cfg, device=device)
        ckpt_path = os.path.join("checkpoints", "baseline_best.pt")
    else:
        cfg = AdvancedDDQNConfig(max_episodes=max_episodes)
        agent = AdvancedDuelingDDQNAgent(env_cfg, cfg, device=device)
        ckpt_path = os.path.join("checkpoints", "advanced_best.pt")

    recent_scores = deque(maxlen=100)
    best_avg_score = -float("inf")

    # Create renderable Gym env
    env = gym.make(env_cfg.env_id, render_mode="human")

    for ep in range(1, max_episodes + 1):
        should_render = (ep % render_every == 0)
        state, _ = env.reset(seed=seed + ep)
        done = False
        ep_reward = 0.0
        steps = 0
        ep_losses = []

        while not done:
            action = agent.select_action(state, evaluate=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            loss = agent.step(state, action, reward, next_state, done)
            if loss is not None:
                ep_losses.append(loss)

            state = next_state
            ep_reward += reward
            steps += 1

        agent.decay_epsilon()
        recent_scores.append(ep_reward)
        cur_avg = float(np.mean(recent_scores))
        mean_loss = float(np.mean(ep_losses)) if ep_losses else 0.0

        if len(recent_scores) >= 15 and cur_avg > best_avg_score:
            best_avg_score = cur_avg
            agent.save(ckpt_path)

        outcome = "LANDED" if ep_reward >= 150 else ("CRASHED" if ep_reward < 0 else "TIMEOUT")
        print(
            f"Ep {ep:3d}/{max_episodes} | "
            f"Result: {outcome:<7} | "
            f"Score: {ep_reward:6.1f} | "
            f"Avg(100): {cur_avg:6.1f} | "
            f"Eps: {agent.epsilon:.3f} | "
            f"Loss: {mean_loss:.4f} | "
            f"Steps: {steps}"
        )

        if cur_avg >= 200.0 and len(recent_scores) >= 50:
            print(f"\n[SOLVED] Lunar Lander Solved at Episode {ep}! 100-Ep Average: {cur_avg:.2f} pts!\n")

    env.close()
    print("\n[COMPLETE] Training session closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Visual Training in Gym")
    parser.add_argument(
        "--model",
        type=str,
        choices=["baseline", "advanced"],
        default="advanced",
        help="Which model to train live",
    )
    parser.add_argument("--episodes", type=int, default=100, help="Number of training episodes")
    parser.add_argument("--render-every", type=int, default=1, help="Render window frequency")
    args = parser.parse_args()

    train_live_visual(model_type=args.model, max_episodes=args.episodes, render_every=args.render_every)
