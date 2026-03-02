"""
Пространство действий Tower of Hanoi.

Action: переместить один из верхних дисков на другую палку.
Действие задаётся парой (from_stick, to_stick), индексы палок 0, 1, 2.
"""


def get_action_space(num_sticks: int) -> list:
    """
    Получить полное пространство действий (все возможные перемещения между палками).
    
    Input:
        num_sticks — количество палок (обычно 3)
    Output: список кортежей [(from_stick, to_stick), ...]
            для 3 палок: (0,1), (0,2), (1,0), (1,2), (2,0), (2,1)
            from_stick != to_stick
    """
    ...


def get_valid_actions(state: tuple, num_sticks: int) -> list:
    """
    Получить список допустимых действий из текущего состояния.
    
    Учитывает:
        — только верхний диск на палке можно брать (тот, у кого height=0)
        — нельзя класть диск на диск меньшего размера
        — from_stick и to_stick должны быть разными
    
    Input:
        state — текущее состояние (sticks, heights)
        num_sticks — количество палок
    Output: список допустимых действий [(from_stick, to_stick), ...]
    """
    ...


def action_to_index(action: tuple, action_space: list) -> int:
    """
    Преобразовать действие в индекс для Q-таблицы или выхода нейросети.
    
    Input:
        action — (from_stick, to_stick)
        action_space — список из get_action_space (порядок должен быть фиксированным)
    Output: int — индекс действия в action_space (0..len(action_space)-1)
    Raises: ValueError если action не найден в action_space
    """
    ...


def index_to_action(index: int, action_space: list) -> tuple:
    """
    Преобразовать индекс в действие.
    
    Input:
        index — индекс в action_space (0..len(action_space)-1)
        action_space — список из get_action_space
    Output: (from_stick, to_stick)
    Raises: IndexError если index вне диапазона
    """
    ...


def is_valid_move(state: tuple, action: tuple) -> bool:
    """
    Проверить, допустим ли ход с точки зрения правил игры.
    
    Правило: нельзя класть диск на диск меньшего размера.
    Также: действие должно брать верхний диск (height=0) и класть на другую палку.
    
    Input:
        state — текущее состояние (sticks, heights)
        action — (from_stick, to_stick)
    Output: True если ход допустим, False иначе
    """
    ...
