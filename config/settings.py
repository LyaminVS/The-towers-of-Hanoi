"""
Конфигурация параметров среды Tower of Hanoi и обучения.

Все параметры можно переопределить через аргументы командной строки
в run/train.py, run/evaluate.py, run/play.py.
"""

# --- Параметры игры ---
NUM_DISKS = 4  # количество дисков (3 — минимум, 5+ — сложнее)
NUM_STICKS = 3  # количество палок (всегда 3 для классической задачи)

# --- Награды и штрафы (env.rewards.Reward) ---
# новая схема: -1 за каждый шаг, штраф -10 когда все диски на последнем стержне,
# и очень большой штраф -50 за попытку положить больший диск на меньший.
REWARD_STEP = -1.0  # за каждый шаг (штраф за длину пути)
REWARD_GOAL = 1000.0  # большой бонус когда все диски правильно на последнем стержне
REWARD_INVALID_MOVE = -50.0  # штраф за недопустимый ход (больший на меньший)
REWARD_CORRECT_PLACEMENT = +1  # убрано: бонус только за финальный успех через REWARD_GOAL
USE_CORRECT_PLACEMENT = True  # применять бонус за правильные диски
# устаревшие параметры
REWARD_DEATH = 0.0
USE_DEATH_PENALTY = False

# --- Параметры обучения ---
LEARNING_RATE = 0.01  # для табличных методов (если появятся)
GAMMA = 0.99  # коэффициент дисконтирования для discounted returns (Σ γ^k * r_k)
DISCOUNT_FACTOR = 0.99  # алиас для GAMMA (для совместимости)
NUM_EPISODES = 5000  # количество эпизодов обучения
MAX_STEPS_PER_EPISODE = 200  # лимит шагов (2^n - 1 для n дисков)
LOG_INTERVAL = 100  # логировать каждые N эпизодов
CHECKPOINT_INTERVAL = 1000  # сохранять чекпоинт каждые N эпизодов
RANDOM_INIT = True  # случайное начальное состояние в каждом эпизоде

# --- Выбор метода агента ---
# "reinforce" | "reinforce_baseline" | "trpo"
AGENT_METHOD = "reinforce"

# --- REINFORCE ---
REINFORCE_LR = 1e-3
REINFORCE_HIDDEN_DIMS = [128, 128]
REINFORCE_ENTROPY_COEF = 0.1  # коэффициент энтропии для исследования  # размеры скрытых слоёв policy

# --- REINFORCE + baseline ---
REINFORCE_BASELINE_LR = 1e-3
REINFORCE_BASELINE_VALUE_LR = 1e-2
REINFORCE_BASELINE_HIDDEN_DIMS = [64, 64]

# --- TRPO ---
TRPO_MAX_KL = 0.01  # лимит KL-дивергенции
TRPO_CG_ITERS = 10
TRPO_BACKTRACK_ITERS = 10
TRPO_BACKTRACK_COEF = 0.5
TRPO_HIDDEN_DIMS = [64, 64]

# --- Параметры оценки ---
EVAL_MODEL_PATH = "model.pth"  # путь к загружаемой модели
EVAL_NUM_EPISODES = 10  # кол-во эпизодов для оценки
EVAL_RENDER = False  # визуализация при оценке
EVAL_SAVE_RESULTS = None  # путь для сохранения результатов (None — не сохранять)
EVAL_PARAMS_FILE = "eval_params.json"  # файл для save/load параметров оценки
EVAL_SAMPLE = True  # если True, во время оценки используется семплирование, иначе argmax

# --- Логирование ---
LOG_FILE = None  # путь к файлу логов (None — только консоль)
LOG_LEVEL = "INFO"  # DEBUG | INFO | WARNING | ERROR
