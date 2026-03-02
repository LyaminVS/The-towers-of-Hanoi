# FILE: ./run/train.py
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings
from env.environment import create_env
from env.rewards import Reward
from env.actions import get_action_space
from agent import create_agent
from training.trainer import train
from training.logger import setup_logger, log_message

def main():
    setup_logger(
        log_file=settings.LOG_FILE,
        level=settings.LOG_LEVEL
    )
    
    reward_scheme = Reward.from_config(settings)
    
    env = create_env(
        num_disks=settings.NUM_DISKS,
        num_sticks=settings.NUM_STICKS,
        max_steps=settings.MAX_STEPS_PER_EPISODE,
        reward=reward_scheme
    )
    
    obs_dim = settings.NUM_DISKS * 2
    action_space = get_action_space(settings.NUM_STICKS)
    
    # Конфиг должен содержать все параметры для выбранного метода
    agent_config = {
        "learning_rate": settings.REINFORCE_LR,
        "discount_factor": settings.DISCOUNT_FACTOR,
        "hidden_dims": settings.REINFORCE_HIDDEN_DIMS,
        # Параметры для baseline если выбран он
        "value_lr": settings.REINFORCE_BASELINE_VALUE_LR,
    }

    log_message(f"Starting: Method={settings.AGENT_METHOD}, Disks={settings.NUM_DISKS}")
    
    agent = create_agent(settings.AGENT_METHOD, obs_dim, action_space, agent_config)
    
    try:
        train(
            env=env,
            agent=agent,
            num_episodes=settings.NUM_EPISODES,
            max_steps_per_episode=settings.MAX_STEPS_PER_EPISODE,
            use_death_penalty=settings.USE_DEATH_PENALTY, # Передаем из настроек
            log_interval=settings.LOG_INTERVAL
        )
        agent.save("model.pth")
        log_message("Training finished. Model saved to model.pth")
        
    except KeyboardInterrupt:
        log_message("Training stopped by user.")

if __name__ == "__main__":
    main()