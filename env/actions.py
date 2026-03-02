"""
Пространство действий Tower of Hanoi.

Action: переместить один из верхних дисков на другую палку.
Действие задаётся парой (from_stick, to_stick), индексы палок 0, 1, 2.

State задается tuple ((stick_0, height_0),  (stick_1, height_1), ...) где нижний индекс соответсвует номеру диска: 0 - самый большой, n-1 - самый маленький

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
    return [(i, j) for i in range(num_sticks) for j in range(num_sticks) if i != j]



def get_valid_actions(state: tuple, num_sticks: int) -> list:
    """
    Получить список допустимых действий из текущего состояния.
    
    Учитывает:
        — stick с которого берут (from_stick) должен быть непустой
        — нельзя класть диск на диск меньшего размера
        — from_stick и to_stick должны быть разными
    
    Input:
        state — текущее состояние ((stick_0, height_0), (stick_1, height_1), ...)
        num_sticks — количество палок
    Output: список допустимых действий [(from_stick, to_stick), ...]
    """
    return [
        (from_stick, to_stick)
        for from_stick in range(num_sticks)
        for to_stick in range(num_sticks)
        if from_stick != to_stick and is_valid_move(state, (from_stick, to_stick))
    ]


def action_to_index(action: tuple, action_space: list) -> int:
    """
    Преобразовать действие в индекс для Q-таблицы или выхода нейросети.
    
    Input:
        action — (from_stick, to_stick)
        action_space — список из get_action_space (порядок должен быть фиксированным)
    Output: int — индекс действия в action_space (0..len(action_space)-1)
    Raises: ValueError если action не найден в action_space
    """
    if action not in action_space:
        raise ValueError(f"Action {action} not found in action_space")
    return action_space.index(action)


def index_to_action(index: int, action_space: list) -> tuple:
    """
    Преобразовать индекс в действие.
    
    Input:
        index — индекс в action_space (0..len(action_space)-1)
        action_space — список из get_action_space
    Output: (from_stick, to_stick)
    Raises: IndexError если index вне диапазона
    """
    if index < 0 or index >= len(action_space):
        raise IndexError(f"Index {index} is out of range for action_space of size {len(action_space)}")
    return action_space[index]


def is_valid_move(state: tuple, action: tuple) -> bool:
    """
    Проверить, допустим ли ход с точки зрения правил игры.
    
    from_stick должен быть не пустой
    
    Input:
        state — текущее состояние ((stick_0, height_0), (stick_1, height_1), ...)
        action — (from_stick, to_stick)
    Output: True если ход допустим, False иначе
    """
    return len([disk_idx for disk_idx, (s, h) in enumerate(state) if s == action[0]]) != 0

