"""
Скрипт игры в Tower of Hanoi вручную через красивый графический интерфейс.
Запуск: python run/play.py --num_disks 3
"""

import sys
import argparse
from pathlib import Path

# Добавляем корень проекта в sys.path, чтобы работали импорты env, utils и т.д.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env.state import get_initial_state, is_terminal_state
from env.actions import get_valid_actions
from env.render import PygameRenderer


def parse_args():
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(description="Play Tower of Hanoi manually.")
    parser.add_argument("--num_disks", type=int, default=3, help="Number of disks to play with (default: 3)")
    return parser.parse_args()


def manual_step(state: tuple, action: tuple) -> tuple:
    """
    Ручное выполнение шага (изменение состояния).
    Переносит верхний диск с from_stick на to_stick.
    Эта функция эмулирует env.step(), пока environment.py не написан.
    """
    from_stick, to_stick = action
    
    # Ищем все диски на from_stick и to_stick
    from_disks = [(idx, h) for idx, (s, h) in enumerate(state) if s == from_stick]
    to_disks = [(idx, h) for idx, (s, h) in enumerate(state) if s == to_stick]
    
    # Находим верхний диск на from_stick
    moving_disk_idx, _ = max(from_disks, key=lambda x: x[1])
    
    # Вычисляем новую высоту на to_stick
    new_height = 0
    if to_disks:
        new_height = max(to_disks, key=lambda x: x[1])[1] + 1

    # Создаем новое состояние
    new_state = list(state)
    new_state[moving_disk_idx] = (to_stick, new_height)
    
    return tuple(new_state)


def main():
    args = parse_args()
    num_disks = args.num_disks
    num_sticks = 3

    print(f"Starting Tower of Hanoi with {num_disks} disks.")
    print("Controls: Click on sticks using mouse, or press 1, 2, 3 on keyboard.")

    # Инициализация состояния и рендерера
    state = get_initial_state(num_disks)
    renderer = PygameRenderer(num_disks=num_disks, num_sticks=num_sticks)
    
    step_count = 0

    try:
        while True:
            # Получаем все доступные ходы для текущего состояния (из actions.py)
            valid_actions = get_valid_actions(state, num_sticks)

            # Передаем управление рендереру (ожидает клика мыши или клавиатуры)
            action = renderer.get_human_action(state, step_count, valid_actions)

            # Если пользователь закрыл окно или нажал ESC
            if action is None:
                print("\nGame exited.")
                break

            # Выполняем действие
            state = manual_step(state, action)
            step_count += 1

            # Проверка на победу (из state.py)
            if is_terminal_state(state, num_disks):
                renderer.render(state, step_count, message="Perfect!")
                renderer.show_victory(step_count)
                print(f"\nVictory! You solved it in {step_count} steps.")
                
                # Минимально возможное число шагов: 2^n - 1
                min_steps = (2 ** num_disks) - 1
                if step_count == min_steps:
                    print("Awesome! You found the optimal path!")
                break

    except KeyboardInterrupt:
        print("\nGame interrupted by user.")
    finally:
        renderer.close()


if __name__ == "__main__":
    main()