# FILE: run_experiments.py
import json
import subprocess
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# --- Setup Paths ---
PROJECT_ROOT = Path.cwd().resolve()
if PROJECT_ROOT.name == "experiments":
    PROJECT_ROOT = PROJECT_ROOT.parent

LOG_DIR = PROJECT_ROOT / "logs"
EXP_DIR = PROJECT_ROOT / "experiments"
LOG_DIR.mkdir(exist_ok=True)
EXP_DIR.mkdir(exist_ok=True)

print(f"--- Starting Experiments ---")
print(f"Project root: {PROJECT_ROOT}")

# --- 1. Training Phase ---

def run_training(method_name, extra_args):
    print(f"\n>>> Running training for: {method_name}...")
    
    base_args = [
        sys.executable, "run/train.py",
        "--num_disks", "4",
        "--num_sticks", "3",
        "--seed", "42",
        "--reward_step", "-2.0",
        "--reward_goal", "4.0",
        "--reward_invalid_move", "0",
        "--reward_correct_placement", "2",
        "--no-use_correct_placement",
        "--num_episodes", "10000",
        "--max_steps", "200",
        "--log_interval", "50",
        "--checkpoint_interval", "1000",
        "--random_init",
        "--no-entropy_adaptive"
    ]
    
    cmd = base_args + extra_args
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    print(f"DONE: {method_name} training finished.")

# REINFORCE
run_training("REINFORCE", [
    "--agent_method", "reinforce",
    "--learning_rate", "0.0001",
    "--entropy_coef", "0.8",
    "--max_grad_norm", "100.0",
    "--save_model", "model_reinforce.pth",
    "--history_path", "logs/training_history_reinforce.json"
])

# REINFORCE + Baseline
run_training("REINFORCE+Baseline", [
    "--agent_method", "reinforce_baseline",
    "--learning_rate", "0.01",
    "--value_lr", "0.01",
    "--entropy_coef", "0.8",
    "--max_grad_norm", "5.0",
    "--save_model", "model_reinforce_baseline.pth",
    "--history_path", "logs/training_history_reinforce_baseline.json"
])

# TRPO
run_training("TRPO", [
    "--agent_method", "trpo",
    "--learning_rate", "0.001",
    "--entropy_coef", "0.2",
    "--episodes_per_update", "10",
    "--trpo_max_kl", "0.05",
    "--trpo_cg_iters", "20",
    "--trpo_backtrack_iters", "15",
    "--trpo_backtrack_coef", "0.5",
    "--trpo_max_abs_advantage", "0.0",
    "--trpo_max_grad_norm_cg", "50.0",
    "--trpo_entropy_coef", "0.1",
    "--trpo_damping", "0.1",
    "--save_model", "model_trpo.pth",
    "--history_path", "logs/training_history_trpo.json"
])

# --- 2. Plotting Phase ---

print("\n--- Generating Comparison Plots ---")

LOG_FILES = {
    "REINFORCE": LOG_DIR / "training_history_reinforce.json",
    "REINFORCE+baseline": LOG_DIR / "training_history_reinforce_baseline.json",
    "TRPO": LOG_DIR / "training_history_trpo.json",
}

def load_history(path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def moving_average(data, window=50):
    if len(data) < window:
        return np.array(data)
    return np.convolve(data, np.ones(window) / window, mode="valid")

histories = {name: load_history(p) for name, p in LOG_FILES.items()}
missing = [n for n, h in histories.items() if h is None]
if missing:
    print(f"Error: Missing log files for {missing}")
    sys.exit(1)

colors = {"REINFORCE": "C0", "REINFORCE+baseline": "C1", "TRPO": "C2"}
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

for name, history in histories.items():
    if not history: continue
    
    rewards = np.array([h["reward"] for h in history])
    success = np.array([h["success"] for h in history], dtype=float)
    steps = np.array([h["steps"] for h in history])

    # Plot 1: Rewards
    ax1.plot(rewards, alpha=0.15, color=colors[name])
    ma_r = moving_average(rewards, 50)
    ax1.plot(np.arange(len(ma_r)) + 25, ma_r, color=colors[name], linewidth=2, label=name)

    # Plot 2: Success Rate
    success_smooth = moving_average(success, 100)
    ax2.plot(np.arange(len(success_smooth)) + 50, success_smooth, color=colors[name], linewidth=2, label=name)

    # Plot 3: Steps
    ax3.plot(steps, alpha=0.15, color=colors[name])
    ma_s = moving_average(steps, 50)
    ax3.plot(np.arange(len(ma_s)) + 25, ma_s, color=colors[name], linewidth=2, label=name)

ax1.set_title("Total Reward per Episode (Smooth)"); ax1.set_ylabel("Reward"); ax1.legend(); ax1.grid(True)
ax2.set_title("Success Rate (Rolling Window 100)"); ax2.set_ylabel("Rate"); ax2.set_ylim(-0.05, 1.05); ax2.legend(); ax2.grid(True)
ax3.set_title("Steps per Episode (Lower is Better)"); ax3.set_ylabel("Steps"); ax3.set_xlabel("Episode"); ax3.legend(); ax3.grid(True)

plt.tight_layout()
out_path = EXP_DIR / "training_curves.png"
plt.savefig(out_path, dpi=120)
print(f"SUCCESS: Comparison chart saved at {out_path}")