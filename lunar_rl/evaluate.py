"""Evaluation and Benchmarking Suite for Lunar Lander RL Models.

Performs:
1. Side-by-side statistical evaluation over 50 test seeds.
2. Performance telemetry comparison (Landing rate, Crash rate, Average score).
3. Video / GIF recording of test rollouts.
"""

import os
import argparse
import numpy as np
import torch
import gymnasium as gym
import imageio
from typing import Dict, Any, List, Optional

from lunar_rl.config import EnvConfig, BaselineDQNConfig, AdvancedDDQNConfig
from lunar_rl.models.baseline_dqn import BaselineDQNAgent
from lunar_rl.models.dueling_ddqn import AdvancedDuelingDDQNAgent


def evaluate_model_stats(
    agent,
    env: gym.Env,
    num_episodes: int = 50,
    base_seed: int = 1000,
) -> Dict[str, Any]:
    """Run comprehensive statistical evaluation over fixed benchmark seeds."""
    rewards = []
    durations = []
    landed_count = 0
    crashed_count = 0

    for i in range(num_episodes):
        state, _ = env.reset(seed=base_seed + i)
        done = False
        ep_reward = 0.0
        steps = 0

        while not done:
            action = agent.select_action(state, evaluate=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            ep_reward += reward
            steps += 1

        rewards.append(ep_reward)
        durations.append(steps)

        if ep_reward >= 180:
            landed_count += 1
        elif ep_reward < 0:
            crashed_count += 1

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "median_reward": float(np.median(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "landing_rate": float(landed_count / num_episodes),
        "crash_rate": float(crashed_count / num_episodes),
        "avg_duration": float(np.mean(durations)),
        "scores": rewards,
    }


def record_rollout_gif(
    agent,
    env_id: str,
    output_path: str,
    seed: int = 42,
    fps: int = 30,
):
    """Record an episode rollout to GIF/MP4 format."""
    env = gym.make(env_id, render_mode="rgb_array")
    state, _ = env.reset(seed=seed)
    frames = []
    done = False
    total_reward = 0.0

    while not done:
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        action = agent.select_action(state, evaluate=True)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
        total_reward += reward

    env.close()

    if frames:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        imageio.mimsave(output_path, frames, fps=fps)
        print(f"[VIDEO] Saved rollout animation to: {output_path} (Return: {total_reward:.1f} pts)")


def run_benchmark(
    checkpoints_dir: str = "checkpoints",
    reports_dir: str = "reports",
    num_episodes: int = 50,
    record_videos: bool = True,
):
    """Run full benchmark comparing Model 1 (Baseline) and Model 2 (Advanced)."""
    env_cfg = EnvConfig()
    eval_env = gym.make(env_cfg.env_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    baseline_ckpt = os.path.join(checkpoints_dir, "baseline_best.pt")
    advanced_ckpt = os.path.join(checkpoints_dir, "advanced_best.pt")

    print("\n=======================================================")
    print("[BENCHMARK] LUNAR LANDER RL - HEAD-TO-HEAD COMPARISON")
    print("=======================================================\n")

    results = {}

    # 1. Evaluate Baseline Model
    if os.path.exists(baseline_ckpt):
        print("Evaluating Model 1 (Vanilla DQN Baseline)...")
        b_cfg = BaselineDQNConfig()
        baseline_agent = BaselineDQNAgent(env_cfg, b_cfg, device=device)
        baseline_agent.load(baseline_ckpt)
        results["Baseline DQN"] = evaluate_model_stats(baseline_agent, eval_env, num_episodes)
        if record_videos:
            record_rollout_gif(
                baseline_agent,
                env_cfg.env_id,
                os.path.join(reports_dir, "rollouts", "baseline_rollout.gif"),
                seed=101,
            )
    else:
        print(f"[WARN] Checkpoint '{baseline_ckpt}' not found. Please train baseline first.")

    # 2. Evaluate Advanced Model
    if os.path.exists(advanced_ckpt):
        print("Evaluating Model 2 (Dueling Double DQN + PER)...")
        a_cfg = AdvancedDDQNConfig()
        advanced_agent = AdvancedDuelingDDQNAgent(env_cfg, a_cfg, device=device)
        advanced_agent.load(advanced_ckpt)
        results["Advanced Dueling DDQN"] = evaluate_model_stats(advanced_agent, eval_env, num_episodes)
        if record_videos:
            record_rollout_gif(
                advanced_agent,
                env_cfg.env_id,
                os.path.join(reports_dir, "rollouts", "advanced_rollout.gif"),
                seed=101,
            )
    else:
        print(f"[WARN] Checkpoint '{advanced_ckpt}' not found. Please train advanced model first.")

    eval_env.close()

    # Print Comparison Table
    if results:
        print("\n" + "=" * 70)
        print(f"{'Metric':<25} | {'Model 1 (Vanilla DQN)':<20} | {'Model 2 (Dueling DDQN)':<20}")
        print("-" * 70)

        m1 = results.get("Baseline DQN", {})
        m2 = results.get("Advanced Dueling DDQN", {})

        def fv(d, k, fmt="{:.1f}"):
            v = d.get(k)
            return fmt.format(v) if v is not None else "N/A"

        print(f"{'Mean Return':<25} | {fv(m1, 'mean_reward'):<20} | {fv(m2, 'mean_reward'):<20}")
        print(f"{'Std Deviation':<25} | {fv(m1, 'std_reward'):<20} | {fv(m2, 'std_reward'):<20}")
        print(f"{'Median Return':<25} | {fv(m1, 'median_reward'):<20} | {fv(m2, 'median_reward'):<20}")
        print(f"{'Max Return':<25} | {fv(m1, 'max_reward'):<20} | {fv(m2, 'max_reward'):<20}")
        print(f"{'Landing Success Rate':<25} | {fv(m1, 'landing_rate', '{:.1%}'):<20} | {fv(m2, 'landing_rate', '{:.1%}'):<20}")
        print(f"{'Crash Rate':<25} | {fv(m1, 'crash_rate', '{:.1%}'):<20} | {fv(m2, 'crash_rate', '{:.1%}'):<20}")
        print(f"{'Avg Episode Steps':<25} | {fv(m1, 'avg_duration'):<20} | {fv(m2, 'avg_duration'):<20}")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Trained Lunar Lander Models")
    parser.add_argument("--episodes", type=int, default=50, help="Number of test episodes")
    parser.add_argument("--no-video", action="store_true", help="Disable GIF recording")
    args = parser.parse_args()

    run_benchmark(num_episodes=args.episodes, record_videos=not args.no_video)
