"""Plotting and Visualization Utilities for Lunar Lander Training Telemetry."""

import os
import json
import matplotlib.pyplot as plt
import numpy as np


def plot_comparison(
    reports_dir: str = "reports",
    output_image: str = "reports/training_comparison.png",
):
    """Generate dual comparison plots for Baseline vs Advanced models."""
    baseline_json = os.path.join(reports_dir, "metrics_baseline.json")
    advanced_json = os.path.join(reports_dir, "metrics_advanced.json")

    has_baseline = os.path.exists(baseline_json)
    has_advanced = os.path.exists(advanced_json)

    if not has_baseline and not has_advanced:
        print("[WARN] No metric JSON files found in reports/. Run train.py first.")
        return

    plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), dpi=150, sharex=True)

    # 1. Rewards & Moving Average Plot
    ax1.set_title("LunarLander-v3: Training Convergence & Episode Returns", fontsize=13, fontweight="bold", pad=10)
    ax1.axhline(200, color="green", linestyle="--", linewidth=1.5, label="Solved Benchmark (+200 pts)")

    if has_baseline:
        with open(baseline_json, "r") as f:
            data_b = json.load(f)
        episodes_b = np.arange(1, len(data_b["scores"]) + 1)
        ax1.plot(episodes_b, data_b["scores"], color="#ff7675", alpha=0.25, label="Baseline Raw Score")
        ax1.plot(episodes_b, data_b["avg_scores"], color="#d63031", linewidth=2.2, label="Model 1: Vanilla DQN (100-Ep Avg)")

    if has_advanced:
        with open(advanced_json, "r") as f:
            data_a = json.load(f)
        episodes_a = np.arange(1, len(data_a["scores"]) + 1)
        ax1.plot(episodes_a, data_a["scores"], color="#74b9ff", alpha=0.25, label="Advanced Raw Score")
        ax1.plot(episodes_a, data_a["avg_scores"], color="#0984e3", linewidth=2.5, label="Model 2: Dueling DDQN + PER (100-Ep Avg)")

    ax1.set_ylabel("Episode Return", fontsize=11)
    ax1.legend(loc="lower right", frameon=True)
    ax1.grid(True, alpha=0.3)

    # 2. Training Loss Trajectory Plot
    ax2.set_title("Training Loss Dynamics & Stability", fontsize=12, fontweight="bold", pad=8)
    all_losses = []
    if has_baseline:
        losses_b = [l for l in data_b["losses"] if l > 0]
        all_losses.extend(losses_b)
        ax2.plot(episodes_b, data_b["losses"], color="#d63031", alpha=0.7, label="Vanilla DQN MSE Loss")
    if has_advanced:
        losses_a = [l for l in data_a["losses"] if l > 0]
        all_losses.extend(losses_a)
        ax2.plot(episodes_a, data_a["losses"], color="#0984e3", alpha=0.7, label="Dueling DDQN Weighted Huber Loss")

    ax2.set_xlabel("Training Episode", fontsize=11)
    ax2.set_ylabel("Loss", fontsize=11)
    if all_losses and any(l > 0 for l in all_losses):
        ax2.set_yscale("log")
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_image), exist_ok=True)
    plt.savefig(output_image, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Training comparison plot successfully exported to: {output_image}")


if __name__ == "__main__":
    plot_comparison()
