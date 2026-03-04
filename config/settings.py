"""
Конфигурация параметров среды Tower of Hanoi и обучения.

Все параметры можно переопределить через аргументы командной строки
в run/train.py, run/evaluate.py, run/play.py.
"""

# --- Параметры игры ---
# 3 диска — проще (оптимум 7 ходов), 4 — сложнее (15 ходов). Для отладки TRPO лучше начать с 3.
NUM_DISKS = 4
NUM_STICKS = 3  # количество палок (всегда 3 для классической задачи)

# --- Награды и штрафы (env.rewards.Reward) ---
# Масштаб -1 за шаг: returns в диапазоне ~[-200, -15], стабильнее для градиентов
REWARD_STEP = -2.0
REWARD_GOAL = 4.0
REWARD_INVALID_MOVE = 0
REWARD_CORRECT_PLACEMENT = 2
USE_CORRECT_PLACEMENT = False

# --- Параметры обучения ---
GAMMA = 0.9  # коэффициент дисконтирования для discounted returns (Σ γ^k * r_k)
DISCOUNT_FACTOR = 0.99  # алиас для GAMMA (для совместимости)
NUM_EPISODES = 20000  # количество эпизодов обучения
MAX_STEPS_PER_EPISODE = 200  # лимит шагов (2^n - 1 для n дисков)
LOG_INTERVAL = 50  # логировать каждые N эпизодов
CHECKPOINT_INTERVAL = 1000  # сохранять чекпоинт каждые N эпизодов
RANDOM_INIT = True  # случайное начальное состояние в каждом эпизоде

# --- Выбор метода агента ---
# "reinforce" | "reinforce_baseline" | "trpo"
AGENT_METHOD = "reinforce_baseline"

# --- REINFORCE (и наследники: baseline, TRPO) ---
REINFORCE_LR = 1e-2  # learning_rate политики — единственный используемый lr в агентах
REINFORCE_HIDDEN_DIMS = [128, 128]
REINFORCE_ENTROPY_COEF = 0.01
REINFORCE_MAX_GRAD_NORM = 5.0

# --- REINFORCE + baseline ---
REINFORCE_BASELINE_LR = 1e-2
REINFORCE_BASELINE_HIDDEN_DIMS = [128, 128]

# --- Батч (TRPO / policy gradient) ---
EPISODES_PER_UPDATE = 10     # эпизодов на одно обновление политики (1 = по одному, 5–10 для TRPO)

# --- TRPO ---
TRPO_MAX_KL = 0.05    # классический trust region (0.01–0.05)
TRPO_CG_ITERS = 20
TRPO_BACKTRACK_ITERS = 15
TRPO_BACKTRACK_COEF = 0.5   # уменьшение шага при отказе line search
TRPO_HIDDEN_DIMS = [128, 128]
TRPO_MAX_ABS_ADVANTAGE = 0.0  # 0 = без клиппинга (нормализация в коде)
TRPO_MAX_GRAD_NORM_CG = 50.0
TRPO_ENTROPY_COEF = 0.2    # умеренная энтропия, без коллапса и без перекоса градиента
TRPO_DAMPING = 0.1          # (H + damping*I) для численной устойчивости CG

# --- Параметры оценки ---
EVAL_MODEL_PATH = "model.pth"  # путь к загружаемой модели
EVAL_NUM_EPISODES = 10  # кол-во эпизодов для оценки
EVAL_RENDER = True  # визуализация при оценке
EVAL_SAVE_RESULTS = None  # путь для сохранения результатов (None — не сохранять)
EVAL_PARAMS_FILE = "eval_params.json"  # файл для save/load параметров оценки
EVAL_SAMPLE = False  # если True, во время оценки используется семплирование, иначе argmax

# --- Логирование ---
LOG_FILE = None  # путь к файлу логов (None — только консоль)
LOG_LEVEL = "INFO"  # DEBUG | INFO | WARNING | ERROR
