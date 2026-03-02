# FILE: ./training/__init__.py
from .trainer import train, run_episode
from .logger import setup_logger, log_episode, log_metrics, log_message

# Если вам действительно нужны там параметры (хотя для обучения они обычно не нужны), 
# теперь импорт будет работать:
from utils.params import save_eval_params, load_eval_params