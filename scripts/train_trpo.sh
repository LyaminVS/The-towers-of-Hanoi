#!/bin/bash
# Обучение TRPO. Все параметры скопированы из config/settings.py, меняется только модель.

set -e
cd "$(dirname "$0")/.."
mkdir -p logs

# ========== ПАРАМЕТРЫ (скопированы из config/settings.py) ==========
# Игра: NUM_DISKS=4, NUM_STICKS=3
NUM_DISKS=4
NUM_STICKS=3

# Награды: REWARD_STEP=-200, REWARD_GOAL=4000, REWARD_INVALID_MOVE=-50,
#          REWARD_CORRECT_PLACEMENT=+100, USE_CORRECT_PLACEMENT=True
REWARD_STEP=-200.0
REWARD_GOAL=4000.0
REWARD_INVALID_MOVE=-50.0
REWARD_CORRECT_PLACEMENT=100
USE_CORRECT_PLACEMENT="--use_correct_placement"

# Обучение: NUM_EPISODES=20000, MAX_STEPS_PER_EPISODE=200, LOG_INTERVAL=50,
#           CHECKPOINT_INTERVAL=1000, RANDOM_INIT=True
NUM_EPISODES=20000
MAX_STEPS=200
LOG_INTERVAL=50
CHECKPOINT_INTERVAL=1000
RANDOM_INIT="--random_init"

# Пути (только модель отличается)
SAVE_MODEL="model_trpo.pth"
HISTORY_PATH="logs/training_history_trpo.json"

# ========== ЗАПУСК ==========
echo "=== Обучение TRPO ==="
echo "  Disks: $NUM_DISKS | Sticks: $NUM_STICKS"
echo "  Эпизодов: $NUM_EPISODES"
echo ""

python run/train.py \
    --num_disks "$NUM_DISKS" \
    --num_sticks "$NUM_STICKS" \
    --agent_method "trpo" \
    --reward_step "$REWARD_STEP" \
    --reward_goal "$REWARD_GOAL" \
    --reward_invalid_move "$REWARD_INVALID_MOVE" \
    --reward_correct_placement "$REWARD_CORRECT_PLACEMENT" \
    $USE_CORRECT_PLACEMENT \
    --num_episodes "$NUM_EPISODES" \
    --max_steps "$MAX_STEPS" \
    --log_interval "$LOG_INTERVAL" \
    --checkpoint_interval "$CHECKPOINT_INTERVAL" \
    $RANDOM_INIT \
    --save_model "$SAVE_MODEL" \
    --history_path "$HISTORY_PATH"

echo ""
echo "=== Готово: $SAVE_MODEL ==="
