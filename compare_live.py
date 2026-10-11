"""Live Comparison Viewer in OpenAI Gym / Gymnasium.

Runs Model 1 (Vanilla DQN) and Model 2 (Dueling DDQN + PER) back-to-back
on the EXACT same initial state seeds in the native Gym desktop window.

Usage:
  python compare_live.py --episodes 10
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


def compare_live(num_trials: int = 5):
    """Run head-to-head comparison in Gym window."""
    env_cfg = EnvConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Model 1
    m1 = BaselineDQNAgent(env_cfg, BaselineDQNConfig(), device=device)
    m1_ckpt = os.path.join("checkpoints", "baseline_best.pt")
    if os.path.exists(m1_ckpt):
        m1.load(m1_ckpt)
        print(f"[LOADED] Model 1 (Vanilla DQN): {m1_ckpt}")
    else:
        print("[WARN] Model 1 checkpoint not found.")

    # Load Model 2
    m2 = AdvancedDuelingDDQNAgent(env_cfg, AdvancedDDQNConfig(), device=device)
    m2_ckpt = os.path.join("checkpoints", "advanced_best.pt")
    if os.path.exists(m2_ckpt):
        m2.load(m2_ckpt)
        print(f"[LOADED] Model 2 (Dueling DDQN + PER): {m2_ckpt}")
    else:
        print("[WARN] Model 2 checkpoint not found.")

    env = gym.make(env_cfg.env_id, render_mode="human")

    print("\n=======================================================")
    print("🚀 LIVE HEAD-TO-HEAD COMPARISON IN GYMNASIUM WINDOW")
    print("=======================================================\n")

    for trial in range(1, num_trials + 1):
        seed = 500 + trial
        print(f"\n================ TRIAL {trial}/{num_trials} (Initial Seed: {seed}) ================")

        # 1. Fly Model 1
        print(">>> Flying Model 1: Vanilla DQN Baseline...")
        s1, _ = env.reset(seed=seed)
        done1 = False
        r1 = 0.0
        steps1 = 0
        while not done1:
            a1 = m1.select_action(s1, evaluate=True)
            s1, rew, term, trunc, _ = env.step(a1)
            done1 = term or trunc
            r1 += rew
            steps1 += 1
            time.sleep(0.015)

        res1 = "LANDED" if r1 >= 150 else ("CRASHED" if r1 < 0 else "TIMEOUT")
        print(f"    Model 1 Result: {res1:<7} | Score: {r1:6.1f} pts | Steps: {steps1}")
        time.sleep(0.8)

        # 2. Fly Model 2
        print(">>> Flying Model 2: Dueling Double DQN + PER...")
        s2, _ = env.reset(seed=seed)
        done2 = False
        r2 = 0.0
        steps2 = 0
        while not done2:
            a2 = m2.select_action(s2, evaluate=True)
            s2, rew, term, trunc, _ = env.step(a2)
            done2 = term or trunc
            r2 += rew
            steps2 += 1
            time.sleep(0.015)

        res2 = "LANDED" if r2 >= 150 else ("CRASHED" if r2 < 0 else "TIMEOUT")
        print(f"    Model 2 Result: {res2:<7} | Score: {r2:6.1f} pts | Steps: {steps2}")
        time.sleep(0.8)

    env.close()
    print("\n[DONE] Comparison complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Head-to-Head Gym Comparison")
    parser.add_argument("--episodes", type=int, default=5, help="Number of comparison trials")
    args = parser.parse_args()

    compare_live(num_trials=args.episodes)
