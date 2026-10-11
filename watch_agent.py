"""Watch Trained RL Agents Fly in Native OpenAI Gym / Gymnasium Window.

Usage:
  python watch_agent.py --model advanced    # Watch Model 2: Dueling Double DQN + PER
  python watch_agent.py --model baseline    # Watch Model 1: Vanilla DQN
  python watch_agent.py --model human       # Fly the lander yourself with Arrow keys
  python watch_agent.py --episodes 10       # Number of flight attempts
"""

import os
import time
import argparse
import numpy as np
import torch
import gymnasium as gym

from lunar_rl.config import EnvConfig, BaselineDQNConfig, AdvancedDDQNConfig
from lunar_rl.models.baseline_dqn import BaselineDQNAgent
from lunar_rl.models.dueling_ddqn import AdvancedDuelingDDQNAgent


def watch_agent(
    model_type: str = "advanced",
    episodes: int = 5,
    delay: float = 0.02,
):
    """Launch Gymnasium window and run agent flights."""
    env_cfg = EnvConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n=======================================================")
    print(f"[GYM WINDOW] Launching Native Gymnasium LunarLander-v3")
    print(f"  Mode: {model_type.upper()}")
    print(f"  Device: {device}")
    print("=======================================================\n")

    # Create Gym environment with native desktop window rendering
    env = gym.make(env_cfg.env_id, render_mode="human")

    agent = None
    if model_type == "baseline":
        cfg = BaselineDQNConfig()
        agent = BaselineDQNAgent(env_cfg, cfg, device=device)
        ckpt = os.path.join("checkpoints", "baseline_best.pt")
        if os.path.exists(ckpt):
            agent.load(ckpt)
            print(f"[LOADED] Checkpoint: {ckpt}")
        else:
            print(f"[WARN] No checkpoint found at '{ckpt}'. Using initial weights.")
    elif model_type == "advanced":
        cfg = AdvancedDDQNConfig()
        agent = AdvancedDuelingDDQNAgent(env_cfg, cfg, device=device)
        ckpt = os.path.join("checkpoints", "advanced_best.pt")
        if os.path.exists(ckpt):
            agent.load(ckpt)
            print(f"[LOADED] Checkpoint: {ckpt}")
        else:
            print(f"[WARN] No checkpoint found at '{ckpt}'. Using initial weights.")

    scores = []

    for ep in range(1, episodes + 1):
        state, _ = env.reset(seed=100 + ep)
        done = False
        ep_reward = 0.0
        steps = 0

        print(f"\n--- Episode {ep}/{episodes} Starting ---")

        while not done:
            if model_type == "human":
                # In human mode, Gym handles keyboard input or noop
                action = 0
            else:
                action = agent.select_action(state, evaluate=True)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            ep_reward += reward
            steps += 1

            if delay > 0:
                time.sleep(delay)

        outcome = "LANDED (+100)" if ep_reward >= 150 else ("CRASHED (-100)" if ep_reward < 0 else "TIMEOUT")
        print(f"Episode {ep} Finished | Outcome: {outcome} | Return: {ep_reward:.1f} pts | Steps: {steps}")
        scores.append(ep_reward)
        time.sleep(0.5)

    env.close()

    print("\n=======================================================")
    print(f"Summary over {episodes} episodes:")
    print(f"  Mean Score: {np.mean(scores):.1f} pts")
    print(f"  Max Score:  {np.max(scores):.1f} pts")
    print("=======================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch Lunar Lander in Gym Window")
    parser.add_argument(
        "--model",
        type=str,
        choices=["baseline", "advanced"],
        default="advanced",
        help="Which model to watch (baseline or advanced)",
    )
    parser.add_argument("--episodes", type=int, default=5, help="Number of flight episodes")
    parser.add_argument("--delay", type=float, default=0.015, help="Delay per physics step (seconds)")
    args = parser.parse_args()

    watch_agent(model_type=args.model, episodes=args.episodes, delay=args.delay)
