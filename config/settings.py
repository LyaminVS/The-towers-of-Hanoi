"""
Конфигурация параметров среды Tower of Hanoi и обучения.

Все параметры можно переопределить через аргументы командной строки
в run/train.py, run/evaluate.py, run/play.py.
"""

# --- Параметры игры ---
NUM_DISKS = 4  # количество дисков (3 — минимум, 5+ — сложнее)
NUM_STICKS = 3  # количество палок (всегда 3 для классической задачи)

# --- Награды и штрафы (env.rewards.Reward) ---
REWARD_STEP = -1.0  # за каждый шаг (штраф за длину пути)
REWARD_INVALID_MOVE = -2.0  # за недопустимый ход (диск на меньший)
REWARD_CORRECT_PLACEMENT = 1.0  # за каждый диск в правильной позиции на 3-й палке
REWARD_DEATH = -100.0  # базовый штраф при смерти (+ штраф за несхоженные шаги)
USE_CORRECT_PLACEMENT = True  # всегда добавлять бонус за correct_placement
USE_DEATH_PENALTY = True  # True — эпизод завершается при invalid, большой штраф

# --- Параметры обучения ---
LEARNING_RATE = 0.05  # для табличных методов (если появятся)
GAMMA = 0.99  # коэффициент дисконтирования для discounted returns (Σ γ^k * r_k)
DISCOUNT_FACTOR = 0.99  # алиас для GAMMA (для совместимости)
NUM_EPISODES = 5000  # количество эпизодов обучения
MAX_STEPS_PER_EPISODE = 100  # лимит шагов (2^n - 1 для n дисков)
LOG_INTERVAL = 100  # логировать каждые N эпизодов
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

# --- Логирование ---
LOG_FILE = None  # путь к файлу логов (None — только консоль)
LOG_LEVEL = "INFO"  # DEBUG | INFO | WARNING | ERROR
