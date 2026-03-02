"""
Представление состояния игры Tower of Hanoi.

State space: [палка каждого диска, высота каждого диска]
Диски нумеруются 0..n-1 (0 — самый маленький, n-1 — самый большой).
"""


def get_initial_state(num_disks: int) -> tuple:
    """
    Создать начальное состояние: все диски на первой палке (индекс 0).
    
    Input:
        num_disks — количество дисков (положительное целое)
    Output: кортеж (sticks, heights)
        sticks — tuple длины num_disks: sticks[i] = индекс палки диска i (0, 1 или 2)
        heights — tuple длины num_disks: heights[i] = высота диска i на своей палке
                 (0 = верхний, 1 = второй сверху, и т.д.)
        Пример для 3 дисков: sticks=(0,0,0), heights=(0,1,2)
    """
    ...


def state_to_observation(state: tuple) -> tuple:
    """
    Преобразовать внутреннее состояние в наблюдение для агента/нейросети.
    
    Input:
        state — внутреннее представление (sticks, heights) из get_initial_state
    Output: observation — формат для нейросети (tuple, list или numpy array)
            размерность должна быть фиксированной для всех состояний
    """
    ...


def observation_to_state(observation) -> tuple:
    """
    Обратное преобразование наблюдения в внутреннее состояние.
    
    Input:
        observation — наблюдение в формате агента (tuple, array)
    Output: state — (sticks, heights), совместимое с get_valid_actions, is_terminal_state
    """
    ...


def is_terminal_state(state: tuple, num_disks: int) -> bool:
    """
    Проверить, достигнута ли цель: все диски на третьей палке.
    
    Input:
        state — текущее состояние (sticks, heights)
        num_disks — количество дисков
    Output: True если все диски на палке с индексом 2 (третья палка), иначе False
    """
    ...


def get_state_hash(state: tuple) -> int | str:
    """
    Получить хешируемое представление состояния (для Q-таблицы или кэша).
    
    Input:
        state — текущее состояние (sticks, heights)
    Output: int или str — значение, пригодное для использования как ключ dict/set
    """
    ...
