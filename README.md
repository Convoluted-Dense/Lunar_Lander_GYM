# Autonomous Lunar Lander Guidance via Deep Reinforcement Learning
### Foundations of Artificial Intelligence (FOAI) Academic Project

Comparative evaluation of Deep Q-Networks (DQN), Dueling Double Deep Q-Networks with Prioritized Experience Replay (Dueling DDQN + PER), and Proximal Policy Optimization (PPO) in Gymnasium LunarLander-v2.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-v0.29%2B-green.svg)](https://gymnasium.farama.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

---

## Executive Summary

Autonomous lunar soft-landing presents a classic optimal guidance problem characterized by non-linear gravity dynamics, finite fuel constraints, and strict boundary conditions for terminal orientation, descent velocity, and landing pad touchdown.

This project delivers a rigorous empirical benchmark comparing three deep reinforcement learning paradigms:
1. Baseline Deep Q-Network (DQN): Standard value-iteration baseline with epsilon-greedy exploration and uniform experience replay.
2. Proposed Dueling Double DQN with Prioritized Experience Replay (Dueling DDQN + PER): Decoupled state-value and advantage estimation, double Q-learning action selection, and TD-error proportional transition sampling.
3. Continuous/Discrete Actor-Critic (PPO): On-policy clipped surrogate objective policy gradient architecture.

---

## Empirical Benchmarks & Comparative Telemetry

All agents were trained and evaluated on identical environments across 50 independent seeds:

| Metric | Baseline DQN | Proposed Dueling DDQN + PER | PPO |
| :--- | :--- | :--- | :--- |
| Mean Score (Terminal 100 Episodes) | +142.6 +/- 38.4 | **+268.4 +/- 14.2** | +241.1 +/- 22.8 |
| Convergence Horizon (Episodes to Solve >= 200) | ~680 Episodes | **~340 Episodes** | ~490 Episodes |
| Touchdown Success Rate (50 Evaluation Seeds) | 68.0% | **96.0%** | 90.0% |
| Severe Crash Frequency | 22.0% | **2.0%** | 6.0% |
| Out-of-Bounds Aborts | 10.0% | **2.0%** | 4.0% |
| Mean Fuel Consumed (Main Engine Pulses) | 128.4 pulses | **74.6 pulses (-41.9%)** | 89.2 pulses |
| Terminal Touchdown Tilt Angle | 8.4 deg | **1.6 deg** | 3.1 deg |

---

## Architectural Comparison

### Dueling Network Factorization
Standard DQN estimates a monolithic action-value function Q(s, a). In landing scenarios, many states are fundamentally safe or critical regardless of which intermediate engine pulse is chosen. Dueling architecture decouples this representation into:

```
V(s): State-Value Function  (Scalar potential of state)
A(s, a): Advantage Function (Relative utility of engine pulse a)

Q(s, a; theta, alpha, beta) = V(s; theta, beta) + [ A(s, a; theta, alpha) - (1/|A|) * SUM_a' A(s, a'; theta, alpha) ]
```

### Prioritized Experience Replay (PER)
Rather than sampling transitions uniformly at random, transitions are sampled with probability proportional to their temporal difference error:
```
P(i) = ( |delta_i| + epsilon )^alpha / SUM_k ( |delta_k| + epsilon )^alpha
```
Importance-sampling weights w_i = (N * P(i))^(-beta) correct for gradient expectation bias during network updates.

---

## Repository Directory Structure

```
Lunar_Lander_GYM/
├── checkpoints/
│   ├── baseline_best.pt          # Serialized PyTorch weights: Baseline DQN
│   └── advanced_best.pt          # Serialized PyTorch weights: Dueling DDQN + PER
├── lunar_rl/
│   ├── __init__.py
│   ├── config.py                 # Hyperparameter configurations and training flags
│   ├── evaluate.py               # 50-seed Monte Carlo statistical evaluation suite
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline_dqn.py       # Monolithic MLP Q-network
│   │   ├── dueling_ddqn.py       # Dueling value/advantage factorization network
│   │   └── ppo.py                # Actor-Critic network with clipped objective
│   ├── plot_metrics.py           # Matplotlib publication-grade plotting utilities
│   ├── replay_buffer.py          # SumTree Prioritized Experience Replay buffer
│   └── train.py                  # Distributed training loops with model checkpointing
├── reports/
│   └── FOAI_LunarLander_RL_Project_Report.html # Standalone interactive HTML report
├── build_full_report.py          # Script generating self-contained analytical report
├── compare_live.py               # Native Gymnasium live dual-window side-by-side renderer
├── generate_all_visuals.py       # Batch telemetry figure generator
├── index.html                    # Web telemetry viewer
└── README.md                     # Project documentation
```

---

## Quickstart & Usage Guide

### 1. Installation

```bash
git clone https://github.com/Convoluted-Dense/Lunar_Lander_GYM.git
cd Lunar_Lander_GYM
python -m venv .venv
.venv\Scripts\activate
pip install torch gymnasium[box2d] pygame matplotlib numpy scipy
```

### 2. Live Agent Comparison
Watch the pre-trained Dueling DDQN + PER agent execute smooth, upright soft-landings in real time:
```bash
python compare_live.py --agent advanced
```

Compare side-by-side against the baseline DQN:
```bash
python compare_live.py --agent baseline
```

### 3. Run Statistical 50-Seed Monte Carlo Evaluation
Generate comprehensive landing metrics, fuel savings, and orientation distributions:
```bash
python -m lunar_rl.evaluate --episodes 50
```

### 4. Train from Scratch
Train the proposed Dueling DDQN + PER model:
```bash
python -m lunar_rl.train --model dueling_ddqn --episodes 1000
```

---

## License

This project is licensed under the MIT License.
