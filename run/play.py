"""
Скрипт игры в Tower of Hanoi вручную (без агента).
Запуск: python run/play.py [--args]
"""


def parse_args() -> object:
    """
    Парсинг аргументов командной строки.
    
    Input: —
    Output: объект с полями:
        num_disks — количество дисков
    """
    ...


def read_move(valid_actions: list) -> tuple:
    """
    Прочитать ход от пользователя (from_stick, to_stick).
    
    Input:
        valid_actions — список допустимых действий для подсказки
    Output: (from_stick, to_stick)
    Raises: KeyboardInterrupt при выходе (Ctrl+C)
    """
    ...


def main() -> None:
    """
    Игровой цикл: создать среду, отображать состояние, читать ходы, step().
    
    Цикл: reset → render → read_move → step → render до terminated/truncated.
    Показывает подсказку допустимых ходов (например "1 2" или "1 3").
    
    Input: —
    Output: None
    """
    ...


if __name__ == "__main__":
    main()
