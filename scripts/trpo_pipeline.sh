#!/bin/bash
# Пайплайн TRPO: фаза 1 (базовое) -> фаза 2 (дообучение)
# Все параметры заданы здесь — меняй по необходимости

set -e
cd "$(dirname "$0")/.."
mkdir -p logs

# ========== ПАРАМЕТРЫ (как в config/settings.py) ==========
# Меняй значения здесь, не трогая settings.py

# --- Игра ---
NUM_DISKS=4
NUM_STICKS=3

# --- Метод агента ---
AGENT_METHOD="trpo"

# --- Награды ---
REWARD_STEP_PHASE1=-1.0
REWARD_GOAL_PHASE1=4000.0
REWARD_INVALID_MOVE_PHASE1=-50.0
REWARD_CORRECT_PLACEMENT_PHASE1=100

REWARD_STEP_PHASE2=-300.0
REWARD_GOAL_PHASE2=4000.0
REWARD_INVALID_MOVE_PHASE2=-50.0
REWARD_CORRECT_PLACEMENT_PHASE2=100

# --- Обучение ---
MAX_STEPS=200
LOG_INTERVAL=100
CHECKPOINT_INTERVAL=1000

# --- Фаза 1 ---
NUM_EPISODES_PHASE1=20000
RANDOM_INIT_PHASE1="--random_init"
ENTROPY_ADAPTIVE_PHASE1="--no-entropy_adaptive"
ENTROPY_COEF_PHASE1=250

# --- Фаза 2 ---
NUM_EPISODES_PHASE2=5000
RANDOM_INIT_PHASE2="--no-random_init"
ENTROPY_ADAPTIVE_PHASE2="--entropy_adaptive"
ENTROPY_COEF_MIN_PHASE2=0
ENTROPY_COEF_MAX_PHASE2=2500
ENTROPY_WINDOW_PHASE2=100

# --- Пути ---
MODEL_PHASE1="model_trpo_phase1.pth"
MODEL_PHASE2="model_trpo_phase2.pth"
HISTORY_PHASE1="logs/training_history_trpo_phase1.json"
HISTORY_PHASE2="logs/training_history_trpo_phase2.json"

# ========== ФАЗА 1 ==========
echo "=== Фаза 1: Базовое обучение (TRPO) ==="
echo "  Disks: $NUM_DISKS | Sticks: $NUM_STICKS | Method: $AGENT_METHOD"
echo "  REWARD_STEP=$REWARD_STEP_PHASE1, RANDOM_INIT=True, entropy_adaptive=False"
echo "  Эпизодов: $NUM_EPISODES_PHASE1"
echo ""

python run/train.py \
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
    --save_model "$MODEL_PHASE1" \
    --history_path "$HISTORY_PHASE1"

# ========== ФАЗА 2 ==========
echo ""
echo "=== Фаза 2: Дообучение (TRPO) ==="
echo "  Disks: $NUM_DISKS | Sticks: $NUM_STICKS | Method: $AGENT_METHOD"
echo "  REWARD_STEP=$REWARD_STEP_PHASE2, RANDOM_INIT=False, entropy_adaptive=True"
echo "  Эпизодов: $NUM_EPISODES_PHASE2"
echo "  Загрузка: $MODEL_PHASE1"
echo ""

python run/train.py \
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
    --save_model "$MODEL_PHASE2" \
    --history_path "$HISTORY_PHASE2"

echo ""
echo "=== Готово ==="
echo "  Модель после фазы 1: $MODEL_PHASE1"
echo "  Модель после фазы 2: $MODEL_PHASE2"
echo "  История фазы 1: $HISTORY_PHASE1"
echo "  История фазы 2: $HISTORY_PHASE2"
