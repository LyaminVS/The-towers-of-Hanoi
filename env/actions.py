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
        — только верхний диск на палке можно брать (тот, у кого height=max(heights(from_stick)) где heights(from_stick) высоты дисков, лежащих на from_stick)
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
    
    Правило: нельзя класть диск на диск меньшего размера.
    Также: действие должно брать верхний диск (height=max(heights(from_stick)) где heights(from_stick) высоты дисков, лежащих на from_stick) и класть на другую палку.
    
    Input:
        state — текущее состояние ((stick_0, height_0), (stick_1, height_1), ...)
        action — (from_stick, to_stick)
    Output: True если ход допустим, False иначе
    """
    from_stick, to_stick = action

    if from_stick == to_stick:
        return False

    # Диски на исходной палке: (индекс_диска, высота)
    # Индекс диска: 0 — самый большой, n-1 — самый маленький
    from_disks = [
        (disk_idx, height)
        for disk_idx, (stick, height) in enumerate(state)
        if stick == from_stick
    ]

    # Нельзя брать с пустой палки
    if not from_disks:
        return False

    # Верхний диск на from_stick — тот, у кого максимальная высота
    moving_disk_idx, _ = max(from_disks, key=lambda x: x[1])

    # Диски на целевой палке
    to_disks = [
        (disk_idx, height)
        for disk_idx, (stick, height) in enumerate(state)
        if stick == to_stick
    ]

    # На пустую палку всегда можно положить
    if not to_disks:
        return True

    # Верхний диск на to_stick
    top_to_disk_idx, _ = max(to_disks, key=lambda x: x[1])

    # Перемещаемый диск должен быть меньше верхнего диска на целевой палке.
    # Меньший диск имеет больший индекс (disk 0 — самый большой).
    return moving_disk_idx > top_to_disk_idx
