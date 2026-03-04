#!/bin/bash
# REINFORCE+baseline Pipeline: phase 1 (basic) -> phase 2 (finetuning)
# All parameters are set here - change as needed

set -e
cd "$(dirname "$0")/.."
mkdir -p logs

# ========== PARAMETERS (like config/settings.py) ==========
# Change values here, don't touch settings.py

# --- Reproducibility ---
SEED=42

# --- Game ---
NUM_DISKS=4
NUM_STICKS=3

# --- Agent method ---
AGENT_METHOD="reinforce"

# --- Rewards ---
REWARD_STEP_PHASE1=-10.0
REWARD_GOAL_PHASE1=4000.0
REWARD_INVALID_MOVE_PHASE1=-50.0
REWARD_CORRECT_PLACEMENT_PHASE1=0

REWARD_STEP_PHASE2=-300.0
REWARD_GOAL_PHASE2=4000.0
REWARD_INVALID_MOVE_PHASE2=-50.0
REWARD_CORRECT_PLACEMENT_PHASE2=100

# --- Training ---
MAX_STEPS=200
LOG_INTERVAL=100
CHECKPOINT_INTERVAL=1000
VALUE_ESTIMATOR="tabular"  # "zero" = no baseline, "tabular" = tabular baseline (N/A for reinforce_baseline)

# --- Gradient Clipping ---
ENABLE_GRAD_CLIPPING=true
MAX_GRAD_NORM=10.0

# --- Phase 1 ---
NUM_EPISODES_PHASE1=10000
RANDOM_INIT_PHASE1="--random_init"
ENTROPY_ADAPTIVE_PHASE1="--no-entropy_adaptive"
ENTROPY_COEF_PHASE1=0

# --- Phase 2 ---
NUM_EPISODES_PHASE2=5000
RANDOM_INIT_PHASE2="--no-random_init"
ENTROPY_ADAPTIVE_PHASE2="--entropy_adaptive"
ENTROPY_COEF_MIN_PHASE2=0
ENTROPY_COEF_MAX_PHASE2=2500
ENTROPY_WINDOW_PHASE2=100

# --- Пути ---
MODEL_PHASE1="model_reinforce_baseline_phase1.pth"
MODEL_PHASE2="model_reinforce_baseline_phase2.pth"
HISTORY_PHASE1="logs/training_history_reinforce_baseline_phase1.json"
HISTORY_PHASE2="logs/training_history_reinforce_baseline_phase2.json"

# Преобразуем ENABLE_GRAD_CLIPPING в MAX_GRAD_NORM
if [ "$ENABLE_GRAD_CLIPPING" = "false" ]; then
  MAX_GRAD_NORM=0
fi

# ========== PHASE 1 ==========
echo "=== Phase 1: Basic training (REINFORCE+baseline) ==="
echo "  Disks: $NUM_DISKS | Sticks: $NUM_STICKS | Method: $AGENT_METHOD | Baseline: $VALUE_ESTIMATOR"
echo "  REWARD_STEP=$REWARD_STEP_PHASE1, RANDOM_INIT=True, entropy_adaptive=False"
echo "  Episodes: $NUM_EPISODES_PHASE1"
echo ""

python run/train.py \
    --seed "$SEED" \
    --num_disks "$NUM_DISKS" \
    --num_sticks "$NUM_STICKS" \
    --agent_method "$AGENT_METHOD" \
    --reward_step "$REWARD_STEP_PHASE1" \
    --reward_goal "$REWARD_GOAL_PHASE1" \
    --reward_invalid_move "$REWARD_INVALID_MOVE_PHASE1" \
    --reward_correct_placement "$REWARD_CORRECT_PLACEMENT_PHASE1" \
    --num_episodes "$NUM_EPISODES_PHASE1" \
    --max_steps "$MAX_STEPS" \
    --log_interval "$LOG_INTERVAL" \
    --checkpoint_interval "$CHECKPOINT_INTERVAL" \
    $RANDOM_INIT_PHASE1 \
    $ENTROPY_ADAPTIVE_PHASE1 \
    --entropy_coef "$ENTROPY_COEF_PHASE1" \
    --max_grad_norm "$MAX_GRAD_NORM" \
    --value_estimator "$VALUE_ESTIMATOR" \
    --save_model "$MODEL_PHASE1" \
    --history_path "$HISTORY_PHASE1"

# ========== PHASE 2 ==========
echo ""
echo "=== Phase 2: Finetuning (REINFORCE+baseline) ==="
echo "  Disks: $NUM_DISKS | Sticks: $NUM_STICKS | Method: $AGENT_METHOD"
echo "  REWARD_STEP=$REWARD_STEP_PHASE2, RANDOM_INIT=False, entropy_adaptive=True"
echo "  Episodes: $NUM_EPISODES_PHASE2"
echo "  Loading: $MODEL_PHASE1"
echo ""

python run/train.py \
    --seed "$SEED" \
    --num_disks "$NUM_DISKS" \
    --num_sticks "$NUM_STICKS" \
    --agent_method "$AGENT_METHOD" \
    --load_model "$MODEL_PHASE1" \
    --reward_step "$REWARD_STEP_PHASE2" \
    --reward_goal "$REWARD_GOAL_PHASE2" \
    --reward_invalid_move "$REWARD_INVALID_MOVE_PHASE2" \
    --reward_correct_placement "$REWARD_CORRECT_PLACEMENT_PHASE2" \
    --num_episodes "$NUM_EPISODES_PHASE2" \
    --max_steps "$MAX_STEPS" \
    --log_interval "$LOG_INTERVAL" \
    --checkpoint_interval "$CHECKPOINT_INTERVAL" \
    $RANDOM_INIT_PHASE2 \
    $ENTROPY_ADAPTIVE_PHASE2 \
    --entropy_coef_min "$ENTROPY_COEF_MIN_PHASE2" \
    --entropy_coef_max "$ENTROPY_COEF_MAX_PHASE2" \
    --entropy_window "$ENTROPY_WINDOW_PHASE2" \
    --max_grad_norm "$MAX_GRAD_NORM" \
    --value_estimator "$VALUE_ESTIMATOR" \
    --save_model "$MODEL_PHASE2" \
    --history_path "$HISTORY_PHASE2"

echo ""
echo "=== Done ==="
echo "  Model after phase 1: $MODEL_PHASE1"
echo "  Model after phase 2: $MODEL_PHASE2"
echo "  History phase 1: $HISTORY_PHASE1"
echo "  History phase 2: $HISTORY_PHASE2"
