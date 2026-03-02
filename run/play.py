"""
Скрипт игры в Tower of Hanoi вручную через красивый графический интерфейс.
Запуск: python run/play.py --num_disks 3
"""

import sys
import argparse
from pathlib import Path

# Добавляем корень проекта в sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env.environment import create_env
from env.actions import get_valid_actions
from env.render import PygameRenderer


def parse_args():
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(description="Play Tower of Hanoi manually.")
    parser.add_argument("--num_disks", type=int, default=3, help="Number of disks (default: 3)")
    return parser.parse_args()


def main():
    args = parse_args()
    num_disks = args.num_disks

    print(f"Starting Tower of Hanoi with {num_disks} disks.")
    print("Controls: Click on sticks using mouse, or press 1, 2, 3 on keyboard.")

    # Используем настоящую среду
    env = create_env(num_disks=num_disks, num_sticks=3)
    renderer = PygameRenderer(num_disks=num_disks, num_sticks=3)
    
    # Первый сброс
    obs, info = env.reset()
    state = info["state"]
    step_count = info["step_count"]
    
    total_reward = 0.0
    message = "Game Started! Make your move."

    try:
        while True:
            # Получаем доступные ходы
            valid_actions = get_valid_actions(tuple(state), env.num_sticks)

            # Рендер и ожидание клика пользователя
            action = renderer.get_human_action(tuple(state), step_count, total_reward, valid_actions, message)

            if action is None:
                print("\nGame exited by user.")
                break

            # Делаем шаг в настоящей среде
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # === ИСПОЛЬЗУЕМ ВСЁ ИЗ INFO ===
            state = info["state"]
            step_count = info["step_count"]
            is_invalid = info["is_invalid"]
            correct_count = info["is_correct_placement"]

            # Формируем динамическое сообщение на основе info от среды
            if is_invalid:
                message = f"Invalid Move! Penalty! (Reward: {reward:.1f})"
            else:
                message = f"Nice! {correct_count}/{num_disks} disks placed correctly. (Reward: {reward:.1f})"

            # Проверка конца эпизода (Победа или смерть от штрафов)
            if terminated or truncated:
                # Если все диски в правильной позиции — это победа
                is_victory = (correct_count == num_disks)
                
                final_msg = "Perfect! Target reached!" if is_victory else "Game Over! (Terminated/Truncated)"
                
                # Последний раз рисуем финальный кадр
                renderer.render(tuple(state), step_count, total_reward, message=final_msg)
                
                # Показываем финальный экран
                renderer.show_end_screen(step_count, total_reward, is_victory)
                
                if is_victory:
                    print(f"\nVictory! Solved in {step_count} steps. Score: {total_reward}")
                else:
                    print(f"\nGame Over. Score: {total_reward}")
                break

    except KeyboardInterrupt:
        print("\nGame interrupted by user (Ctrl+C).")
    finally:
        renderer.close()


if __name__ == "__main__":
    main()