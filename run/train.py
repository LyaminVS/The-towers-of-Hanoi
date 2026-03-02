# FILE: ./run/train.py
import sys
import argparse
from pathlib import Path

# Добавляем корень проекта в путь
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings
from env.environment import create_env
from env.rewards import Reward
from env.actions import get_action_space
from agent import create_agent
from training.trainer import train
from training.logger import setup_logger, log_message
from utils.params import save_history

def main():
    # 1. Настройка логгера (из settings.py)
    setup_logger(
        log_file=settings.LOG_FILE,
        level=settings.LOG_LEVEL
    )
    
    log_message(f"=== Starting Training Session ===")
    log_message(f"Method: {settings.AGENT_METHOD} | Disks: {settings.NUM_DISKS}")

    # 2. Инициализация среды
    reward_scheme = Reward.from_config(settings)
    env = create_env(
        num_disks=settings.NUM_DISKS,
        num_sticks=settings.NUM_STICKS,
        max_steps=settings.MAX_STEPS_PER_EPISODE,
        reward=reward_scheme
    )
    
    # 3. Настройка агента
    obs_dim = settings.NUM_DISKS * 2
    action_space = get_action_space(settings.NUM_STICKS)
    
    # Собираем конфиг агента из всех доступных параметров settings.py
    agent_config = {
        "learning_rate": settings.REINFORCE_LR,
        "discount_factor": settings.DISCOUNT_FACTOR,
        "gamma": settings.GAMMA,
        "hidden_dims": settings.REINFORCE_HIDDEN_DIMS,
        "entropy_coef": getattr(settings, "REINFORCE_ENTROPY_COEF", 0.01),
        "value_lr": getattr(settings, "REINFORCE_BASELINE_VALUE_LR", 1e-2),
        "max_kl": getattr(settings, "TRPO_MAX_KL", 0.01),
    }

    agent = create_agent(settings.AGENT_METHOD, obs_dim, action_space, agent_config)
    
    # 4. Запуск обучения
    try:
        history = train(
            env=env,
            agent=agent,
            num_episodes=settings.NUM_EPISODES,
            max_steps_per_episode=settings.MAX_STEPS_PER_EPISODE,
            use_death_penalty=settings.USE_DEATH_PENALTY,
            log_interval=settings.LOG_INTERVAL,
            random_init=getattr(settings, "RANDOM_INIT", True),
        )
        
        # 5. Сохранение результатов
        # Сохраняем веса нейросети
        model_path = "model.pth"
        agent.save(model_path)
        log_message(f"Model weights saved to {model_path}")
        
        # Сохраняем историю для графиков
        history_path = "logs/training_history.json"
        save_history(history_path, history)
        log_message(f"Training history saved to {history_path}")
        
        log_message("=== Training Successfully Finished ===")
        
    except KeyboardInterrupt:
        log_message("Training interrupted by user. Metrics and model were NOT saved.", level="WARNING")
    except Exception as e:
        log_message(f"Critical error during training: {e}", level="ERROR")
        raise e

if __name__ == "__main__":
    main()