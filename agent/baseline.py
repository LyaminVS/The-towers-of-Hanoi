"""
Классы baseline V(s) для policy gradient методов.

Используется в REINFORCE и TRPO. Все классы реализуют единый интерфейс:
    add_trajectory(episode_traj)  — добавить эпизод в историю
    predict(observations)         — вернуть V(s_t) для списка наблюдений

Доступные реализации:
    ZeroBaseline    — V(s) = 0 всегда (vanilla REINFORCE без baseline)
    TabularBaseline — V(s) по истории последних N траекторий (инкрементальное среднее)

Оценка V(s) в TabularBaseline:
    V(s) = 0, c(s) = 0 для всех s
    for trajectory in trajectory_history:
        for (s, G) in trajectory:
            V(s) += (G - V(s)) / (c(s) + 1)
            c(s) += 1
"""

from collections import deque

import numpy as np

from env.state import observation_to_state


def state_to_index(state: tuple, num_sticks: int = 3) -> int:
    """
    Состояние → уникальный целочисленный индекс.

    State[i] = (stick, height). Индекс определяется только stick[i] —
    высоты однозначно выводятся из назначений палок.

    Index = stick[0] * num_sticks^0 + stick[1] * num_sticks^1 + ...
    (смешанная система счисления по основанию num_sticks)
    """
    index = 0
    base = 1
    for stick, _height in state:
        index += stick * base
        base *= num_sticks
    return index


def index_to_state(index: int, num_disks: int, num_sticks: int = 3) -> tuple:
    """
    Целочисленный индекс → состояние.

    Восстанавливает stick-назначения, затем вычисляет высоты:
        height[i] = кол-во дисков j < i на той же палке
    (диски с меньшим индексом больше и лежат ниже).
    """
    sticks = []
    remaining = index
    for _ in range(num_disks):
        sticks.append(remaining % num_sticks)
        remaining //= num_sticks

    heights = [
        sum(1 for j in range(i) if sticks[j] == sticks[i])
        for i in range(num_disks)
    ]
    return tuple((sticks[i], heights[i]) for i in range(num_disks))


class ZeroBaseline:
    """
    Тривиальный baseline: V(s) = 0 для любого состояния.

    Эквивалентен отсутствию baseline (vanilla REINFORCE).
    Реализует тот же интерфейс, что и TabularBaseline, что позволяет
    использовать один код пути для всех вариантов baseline.
    """

    def add_trajectory(self, episode_traj: list) -> None:
        """No-op: ZeroBaseline не хранит историю."""

    def predict(self, observations: list) -> np.ndarray:
        """
        Возвращает нули для всех наблюдений.

        Input:
            observations — list наблюдений любого формата
        Output: np.ndarray нулей формы (len(observations),)
        """
        return np.zeros(len(observations), dtype=np.float64)


class TabularBaseline:
    """
    Табличная оценка V(s) по истории последних history_len траекторий.

    Таблица проиндексирована состояниями: V[state_to_index(s)] = V̂(s).

    Использование:
        baseline = TabularBaseline(num_disks, num_sticks, history_len)
        baseline.add_trajectory([(state_tuple, G_t), ...])
        values = baseline.predict(observations)
    """

    def __init__(self, num_disks: int, num_sticks: int, history_len: int):
        self.num_disks = num_disks
        self.num_sticks = num_sticks
        self.trajectory_history: deque = deque(maxlen=history_len)

    def add_trajectory(self, episode_traj: list) -> None:
        """
        Добавить эпизод в историю.

        Input:
            episode_traj — list[(state_tuple, G_t)] для одного эпизода
        Output: None
        """
        self.trajectory_history.append(episode_traj)

    def estimate_value_table(self) -> np.ndarray:
        """
        Оценка V(s) по истории последних history_len траекторий.

        Input: —
        Output: np.ndarray формы (num_sticks^num_disks,), V[idx] = V̂(s)
        """
        num_states = self.num_sticks ** self.num_disks
        V = np.zeros(num_states, dtype=np.float64)
        c = np.zeros(num_states, dtype=np.int64)

        for trajectory in self.trajectory_history:
            for state_tuple, G in trajectory:
                idx = state_to_index(state_tuple, self.num_sticks)
                V[idx] += (G - V[idx]) / (c[idx] + 1)
                c[idx] += 1

        return V

    def predict(self, observations: list) -> np.ndarray:
        """
        V(s) для каждого наблюдения из списка.

        Пересчитывает таблицу V по текущей истории, затем
        возвращает V(s_t) для каждого observation.

        Input:
            observations — list наблюдений (формат env)
        Output: np.ndarray формы (len(observations),)
        """
        V = self.estimate_value_table()
        values = np.zeros(len(observations), dtype=np.float64)
        for i, obs in enumerate(observations):
            state = observation_to_state(obs)
            values[i] = V[state_to_index(state, self.num_sticks)]
        return values
