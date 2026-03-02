"""
Конфигурация параметров среды Tower of Hanoi и обучения.

Все параметры можно переопределить через аргументы командной строки
в run/train.py и run/evaluate.py.
"""

# --- Параметры игры ---
NUM_DISKS = 3  # количество дисков
NUM_STICKS = 3  # количество палок

# --- Награды и штрафы (см. env.rewards.Reward) ---
REWARD_STEP = -1.0  # за каждый шаг
REWARD_INVALID_MOVE = -10.0  # за размещение диска на меньший
REWARD_CORRECT_PLACEMENT = 10.0  # за правильное размещение на третьей палке
REWARD_DEATH = -100.0  # при use_death_penalty
USE_CORRECT_PLACEMENT = True  # выдавать бонус за correct_placement
USE_DEATH_PENALTY = False  # завершать эпизод при invalid

# --- Параметры обучения ---
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
NUM_EPISODES = 1000
MAX_STEPS_PER_EPISODE = 100

# --- Выбор метода агента ---
# "reinforce" | "reinforce_baseline" | "trpo"
AGENT_METHOD = "reinforce"

# --- REINFORCE ---
REINFORCE_LR = 1e-3
REINFORCE_HIDDEN_DIMS = 64

# --- REINFORCE + baseline ---
REINFORCE_BASELINE_LR = 1e-3
REINFORCE_BASELINE_VALUE_LR = 1e-2
REINFORCE_BASELINE_HIDDEN_DIMS = 64

# --- TRPO ---
TRPO_MAX_KL = 0.01
TRPO_CG_ITERS = 10
TRPO_BACKTRACK_ITERS = 10
TRPO_BACKTRACK_COEF = 0.5
TRPO_HIDDEN_DIMS = 64

# --- Параметры оценки ---
EVAL_MODEL_PATH = "model.pth"  # путь к загружаемой модели
EVAL_NUM_EPISODES = 10  # кол-во эпизодов для оценки
EVAL_RENDER = False  # визуализация при оценке
EVAL_SAVE_RESULTS = None  # путь для сохранения результатов (None — не сохранять)
EVAL_PARAMS_FILE = "eval_params.json"  # файл для save/load параметров оценки
