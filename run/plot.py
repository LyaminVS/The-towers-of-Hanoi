# FILE: ./run/plot.py
import json
import matplotlib.pyplot as plt
import numpy as np
import os

def moving_average(data, window=50):
    """Сглаживание графика для лучшей читаемости."""
    if len(data) < window: return data
    return np.convolve(data, np.ones(window)/window, mode='valid')

def plot_training_results(history_path="training_history.json"):
    if not os.path.exists(history_path):
        print(f"Error: {history_path} not found. Run train.py first.")
        return

    with open(history_path, 'r') as f:
        history = json.load(f)

    episodes = [h['episode'] for h in history]
    rewards = [h['reward'] for h in history]
    steps = [h['steps'] for h in history]
    success = [h['success'] for h in history]
    
    # Создаем фигуру с 3 графиками
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

    # 1. График наград
    ax1.plot(rewards, alpha=0.3, color='blue', label='Raw Reward')
    ax1.plot(moving_average(rewards), color='darkblue', linewidth=2, label='Smoothed')
    ax1.set_title("Total Reward per Episode")
    ax1.set_ylabel("Reward")
    ax1.legend()
    ax1.grid(True)

    # 2. График успеха (Success Rate)
    success_smooth = moving_average(success, window=100)
    ax2.plot(success_smooth, color='green', linewidth=2)
    ax2.set_title("Success Rate (Rolling Window 100)")
    ax2.set_ylabel("Rate (0.0 to 1.0)")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True)

    # 3. График шагов
    ax3.plot(steps, alpha=0.3, color='orange')
    ax3.plot(moving_average(steps), color='red', linewidth=2)
    ax3.set_title("Steps per Episode (Lower is Better)")
    ax3.set_ylabel("Steps")
    ax3.set_xlabel("Episode")
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig("plots/training_curves.png")
    print("Graphs saved to training_curves.png")
    plt.show()

if __name__ == "__main__":
    plot_training_results()