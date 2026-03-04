import sys
import argparse
from pathlib import Path

# Добавляем корень проекта
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Импорт конфига
from config import settings
from env.environment import create_env
from env.actions import get_valid_actions
from env.render import PygameRenderer
from env.rewards import Reward

def parse_args():
    parser = argparse.ArgumentParser(description="Play Tower of Hanoi.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility (default: 42)")
    # По умолчанию берем значения из settings.py
    parser.add_argument("--num_disks", type=int, default=settings.NUM_DISKS, help="Number of disks")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Установка seed для воспроизводимости
    import numpy as np
    import torch
    import random
    
    if args.seed is not None:
        settings.SEED = args.seed
    
    seed = settings.SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # 1. Создаем схему наград напрямую из конфига
    reward_scheme = Reward.from_config(settings)
    
    # 2. Создаем среду с параметрами из конфига/аргументов
    env = create_env(
        num_disks=args.num_disks, 
        num_sticks=settings.NUM_STICKS, 
        max_steps=settings.MAX_STEPS_PER_EPISODE,
        reward=reward_scheme
    )
    
    renderer = PygameRenderer(num_disks=args.num_disks, num_sticks=settings.NUM_STICKS)
    
    obs, info = env.reset()
    state = info["state"]
    total_reward = 0.0
    message = f"Goal: Move {args.num_disks} disks to the last stick!"

    try:
        while True:
            # Получаем все возможные пары ходов (даже потенциально невалидные для RL)
            valid_actions = [(f, t) for f in range(env.num_sticks) for t in range(env.num_sticks) if f != t]

            action = renderer.get_human_action(tuple(state), env._step_count, total_reward, valid_actions, message)
            if action is None: break

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            state = info["state"]

            # Логика сообщений из info
            if info.get("is_invalid"):
                message = f"Invalid move! Penalty: {reward:.1f}"
            else:
                correct = info.get("is_correct_placement", 0)
                message = f"Step Reward: {reward:.1f} | Placed: {correct}/{args.num_disks}"

            if terminated or truncated:
                is_win = (info.get("is_correct_placement", 0) == args.num_disks)
                renderer.render(tuple(state), env._step_count, total_reward, message="Finished!")
                renderer.show_end_screen(env._step_count, total_reward, is_win)
                break

    except KeyboardInterrupt:
        pass
    finally:
        renderer.close()

if __name__ == "__main__":
    main()