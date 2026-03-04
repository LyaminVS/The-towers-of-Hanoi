
# 🚀 Tower of Hanoi RL: Deployment & Execution Guide

This guide provides a comprehensive walkthrough to set up the environment using Docker and run Reinforcement Learning experiments (REINFORCE, REINFORCE+Baseline, and TRPO).

---

## 📋 Prerequisites

Before starting, ensure you have the following installed on your host machine:
- **Docker**
- **Git**

---

## 🛠 Step 1: Clone the Repository

Clone the project to your local machine and navigate into the project directory:

```bash
git clone https://github.com/LyaminVS/The-towers-of-Hanoi
cd ./The-towers-of-Hanoi
```

---

## 🐳 Step 2: Docker Environment Setup

We use Docker to ensure all dependencies (PyTorch 2.0+, Python 3.10+, CUDA 11.8) are consistent.

### 2.1 Configuration
The `credentials` file maps your local user ID to the container to avoid file permission issues. Check it if you need to change the container name:
```bash
cat credentials
```

### 2.2 Build the Image
Build the Docker image (this takes 2-5 minutes as it installs Conda and PyTorch):
```bash
chmod +x build.sh
./build.sh
```

### 2.3 Launch the Container
Start the container. This script mounts your current directory to `/app` inside the container:
```bash
chmod +x launch_container.sh
./launch_container.sh
```
*Once launched, you will be automatically logged into the container's bash shell.*

---

## 🏃 Step 3: Running Experiments

You have two ways to run the experiments: via the **Command Line** (automated) or **Jupyter Notebook** (interactive).

### Option A: Command Line (Recommended)
We have provided an automated script that trains all three agents with a **fixed seed (42)** for reproducibility and generates comparison plots.

Inside the container shell, run:
```bash
chmod +x run.sh
./run.sh
```
**What this script does:**
1.  Trains **REINFORCE** (10,000 episodes).
2.  Trains **REINFORCE + Baseline** (10,000 episodes).
3.  Trains **TRPO** (10,000 episodes).
4.  Saves logs to `logs/`.
5.  Generates a comparison chart in `experiments/training_curves.png`.

### Option B: Jupyter Notebook
If you prefer running experiments cell-by-cell or visualizing data interactively:

1.  When you run `./launch_container.sh`, a Jupyter server starts automatically on port **8890**.
2.  Open your browser and go to: `http://localhost:8890`
3.  Navigate to the `experiments/` folder.
4.  Open the `.ipynb` notebook.
5.  **Note:** Ensure your notebook uses the same parameters (seed=42) for consistent results.

---

## 📊 Step 4: Visualizing Results

After the training is complete, you can find the results in the following locations:

-   **Comparison Plots:** `experiments/training_curves.png`
    -   *Top Plot:* Smoothed Total Reward.
    -   *Middle Plot:* Success Rate (Rolling window of 100).
    -   *Bottom Plot:* Average Steps per Episode.
-   **Logs:** `logs/training_history_*.json` (Raw data for each method).
-   **Model Checkpoints:** `model_reinforce.pth`, `model_trpo.pth`, etc. (Saved weights).

---

## 🕹 Step 5: Manual Execution & Testing

If you want to run a specific agent manually or evaluate a trained model:

**Manual Training:**
```bash
python run/train.py --agent_method trpo --num_disks 4 --num_episodes 10000 --seed 42
```

**Evaluation (Visualization):**
To see the trained agent actually playing the game:
```bash
# Note: Requires a display/X11 setup if running inside Docker
python run/evaluate.py --load_model model_trpo.pth --num_disks 4 --render
```

**Manual Play (Human):**
Test the game rules yourself:
```bash
python run/play.py --num_disks 4
```

---

## 📂 Project Structure Summary

-   `run/`: Entry point scripts (`train.py`, `evaluate.py`, `plot_comparison.py`).
-   `agent/`: Implementation of RL algorithms.
-   `env/`: Logic for the Tower of Hanoi environment and Pygame rendering.
-   `config/`: Hyperparameters and global settings.
-   `logs/`: JSON data from training sessions.
-   `experiments/`: Visualization outputs and notebooks.

---
**Happy Training!** 🚀