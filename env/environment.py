"""
Среда Tower of Hanoi (Gym-like интерфейс).

State space: list[(stick, height), ...] — для каждого диска
Action space: переместить один из верхних дисков на другую палку
Диск 0 — самый большой (внизу), высота 0 — снизу.
"""

from .rewards import Reward
from .state import get_initial_state, get_random_valid_state, state_to_observation
from .actions import is_valid_move, get_valid_actions as _get_valid_actions, get_action_space as _get_action_space


class TowerOfHanoiEnv:
    """
    Среда «Ханойская башня» для RL.
    
    Атрибуты:
        num_disks: int — количество дисков
        num_sticks: int — количество палок (всегда 3)
        max_steps: int — лимит шагов на эпизод (для truncated)
        reward: Reward — схема наград (настраиваемая)
    """

    def __init__(
        self,
        num_disks: int = 3,
        num_sticks: int = 3,
        max_steps: int = 100,
        reward: Reward | None = None,
    ):
        """
        Инициализация среды.
        
        Input:
            num_disks — количество дисков (по умолчанию 3)
            num_sticks — количество палок (обычно 3)
            max_steps — максимальное число шагов за эпизод; при превышении
                       step() вернёт truncated=True
            reward — экземпляр Reward; если None, создаётся с дефолтными значениями
        Output: —
        """
        # Параметры среды
        self.num_disks = num_disks
        self.num_sticks = num_sticks
        self.max_steps = max_steps
        self.reward = reward if reward is not None else Reward()
        # Внутреннее состояние: список (stick, height) для каждого диска
        self._state: list = []
        self._step_count: int = 0

    def reset(self, random_init: bool = False) -> tuple:
        """
        Сброс среды в начальное состояние.
        
        Input:
            random_init — если True, случайное валидное состояние; иначе все диски на палке 0
        Output: (observation, info)
            observation — начальное наблюдение в формате для агента
            info — dict: state (внутр. состояние), step_count (0), и др.
        """
        if random_init:
            self._state = list(get_random_valid_state(self.num_disks, self.num_sticks))
        else:
            self._state = list(get_initial_state(self.num_disks))
        self._step_count = 0
        observation = state_to_observation(tuple(self._state))
        info = {"state": self._state.copy(), "step_count": self._step_count}
        return observation, info

    def step(self, action: tuple) -> tuple:
        """
        Выполнить действие и перейти в новое состояние.
        
        Input:
            action — (from_stick, to_stick), индексы палок 0..2
        Output: (observation, reward, terminated, truncated, info)
            observation — новое наблюдение после хода
            reward — награда (по схеме Reward: step, invalid, correct)
            terminated — True если цель достигнута (все на 3-й палке)
            truncated — True если step_count >= max_steps
            info — dict: is_invalid, is_correct_placement, step_count, state, ...
        """
        # Проверка допустимости хода (from_stick не пустой)
        state_tuple = tuple(self._state)
        is_invalid = False
        if not is_valid_move(state_tuple, action):
            # попытка хода с пустой палки — считается invalid, состояние не меняется
            is_invalid = True
        else:
            # Применение хода на копии состояния
            from_stick, to_stick = action
            from_disks = [
                (i, self._state[i][1])
                for i in range(self.num_disks)
                if self._state[i][0] == from_stick
            ]
            moving_disk_idx, _ = max(from_disks, key=lambda x: x[1])
            to_disks = [
                (i, self._state[i][1])
                for i in range(self.num_disks)
                if self._state[i][0] == to_stick
            ]
            new_height = (max(h for _, h in to_disks) + 1) if to_disks else 0

            candidate = [list(p) for p in self._state]
            candidate[moving_disk_idx] = [to_stick, new_height]
            new_state = [tuple(p) for p in candidate]

            # Проверка недопустимости: большой диск на меньший
            if self.is_invalid_state(new_state):
                is_invalid = True
                # не обновляем self._state, остаёмся в прежнем состоянии
            else:
                self._state = new_state

        # Обновление счётчика и вычисление награды
        self._step_count += 1
        is_correct_placement = self.get_correct_placement(self._state)
        goal_reached = all(is_correct_placement)
        correct_count = sum(is_correct_placement)
        reward, reward_done = self.reward.compute(
            is_invalid,
            goal_reached,
            correct_count=correct_count,
        )
        # Завершение: цель достигнута или лимит шагов
        terminated = goal_reached or reward_done
        truncated = self._step_count >= self.max_steps

        observation = state_to_observation(tuple(self._state))
        info = {
            "state": self._state.copy(),
            "step_count": self._step_count,
            "is_invalid": is_invalid,
            "is_correct_placement": sum(is_correct_placement),
        }
        return observation, reward, terminated, truncated, info

    def get_correct_placement(self, state: list) -> list[bool]:
        """
        Сравнение состояния с целевым: все диски на последней палке в правильном порядке.
        
        Правильное состояние: на последней палке (индекс num_sticks-1) все диски,
        диск 0 (самый большой) снизу (height=0), диск 1 выше (height=1), и т.д.
        Высота измеряется снизу вверх: 0 — нижний блок.
        
        Input: state — list[(stick, height), ...]
        Output: list[bool] — для каждого диска i: True если он в правильной позиции
        """
        result = [False] * self.num_disks
        target_stick = self.num_sticks - 1
        # tower[height] = disk_id — какие диски на целевой палке и на каких высотах
        tower: dict[int, int] = {}
        for i in range(self.num_disks):
            stick, height = state[i]
            if stick == target_stick:
                tower[height] = i
        if not tower:
            return result
        # Правильный порядок: на высоте h должен быть диск h (0 снизу, 1 выше, ...)
        for h in range(max(tower.keys()) + 1):
            if h not in tower or tower[h] != h:
                break
            result[h] = True
        return result

    def is_invalid_state(self, state: list) -> bool:
        """
        Проверить, что состояние недопустимо: есть диск на меньшем на какой-то палке.
        Диск 0 — самый большой, высота 0 — снизу.
        
        Input: state — list[(stick, height), ...]
        Output: True если состояние недопустимо (есть диск на меньшем), False если допустимо
        """
        # Для каждой палки: диски снизу вверх должны идти по возрастанию размера (0 — самый большой)
        for stick in range(self.num_sticks):
            on_stick = [(i, state[i][1]) for i in range(self.num_disks) if state[i][0] == stick]
            on_stick.sort(key=lambda x: x[1])
            disk_ids = [d[0] for d in on_stick]
            for j in range(len(disk_ids) - 1):
                if disk_ids[j] > disk_ids[j + 1]:
                    return True  # Больший диск лежит на меньшем — недопустимо
        return False


def create_env(
    num_disks: int = 3,
    num_sticks: int = 3,
    max_steps: int = 100,
    reward: "Reward | None" = None,
) -> TowerOfHanoiEnv:
    """
    Фабрика: создать экземпляр среды.
    
    Input:
        num_disks, num_sticks, max_steps — параметры среды
        reward — опционально, экземпляр Reward
    Output: экземпляр TowerOfHanoiEnv
    """
    return TowerOfHanoiEnv(
        num_disks=num_disks,
        num_sticks=num_sticks,
        max_steps=max_steps,
        reward=reward,
    )
