"""
REINFORCE с baseline — уменьшение дисперсии через вычитание V(s).
Наследуется от REINFORCEAgent; добавляет табличный baseline по истории траекторий.

Baseline: табличная оценка V(s) по истории последних N траекторий.
Таблица проиндексирована состояниями: V[state_to_index(s)] = V̂(s).

Оценка V(s) перед каждым градиентным шагом:
    1. V(s) = 0, c(s) = 0 для всех s
    2. for trajectory in trajectory_history:
           for (s, G) in trajectory:
               V(s) += (G - V(s)) / (c(s) + 1)
               c(s) += 1
"""

from collections import deque

import numpy as np
import torch

from .reinforce import REINFORCEAgent
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


class REINFORCEBaselineAgent(REINFORCEAgent):
    """
    REINFORCE + табличный baseline по истории траекторий:
        ∇J(θ) = E[Σ_t (G_t - V(s_t)) ∇ log π(a|s)]

    Наследует от REINFORCEAgent: select_action, store_transition, reset_trajectory,
    _compute_returns, _compute_policy_loss_and_entropy. Переопределяет только update().

    V(s) — таблица размером num_sticks^num_disks.
    Перед каждым градиентным шагом пересчитывается по последним
    history_len эпизодам инкрементальным средним.
    """

    def __init__(self, observation_dim: int, action_space: list, config: dict):
        super().__init__(observation_dim, action_space, config)

        self.num_disks = observation_dim // 2
        self.num_sticks = config.get("num_sticks", 3)
        history_len = config.get("history_len", 20)

        # Кольцевой буфер последних history_len траекторий.
        # Каждая траектория — list[(state_tuple, G_t)].
        self.trajectory_history: deque = deque(maxlen=history_len)

    def _predict_baseline(self, observations: list, V: np.ndarray) -> np.ndarray:
        """V(s) для каждого наблюдения из сохранённой траектории."""
        values = np.zeros(len(observations), dtype=np.float64)
        for i, obs in enumerate(observations):
            state = observation_to_state(obs)
            values[i] = V[state_to_index(state, self.num_sticks)]
        return values

    def _estimate_value_table(self) -> np.ndarray:
        """
        Оценка V(s) по истории последних history_len траекторий.

        V(s) = 0, c(s) = 0 в начале каждого вызова.
        for trajectory in trajectory_history:
            for (s, G) in trajectory:
                V(s) += (G - V(s)) / (c(s) + 1)
                c(s) += 1
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

    def update(self) -> dict:
        if not self.saved_states:
            return {"policy_loss": 0.0, "baseline_mse": 0.0, "mean_return": 0.0}

        device = next(self.policy_network.parameters()).device
        returns = self._compute_returns()

        # 1. Добавляем текущий эпизод в историю
        episode_traj = [
            (observation_to_state(self.saved_states[t]), float(returns[t]))
            for t in range(len(self.saved_states))
        ]
        self.trajectory_history.append(episode_traj)

        # 2. Оцениваем V(s) по истории последних N эпизодов
        V = self._estimate_value_table()

        # 3. Advantages = G_t - V(s_t)
        baseline_values = self._predict_baseline(self.saved_states, V)
        returns_np = np.array(returns, dtype=np.float64)
        advantages = returns_np - baseline_values
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=device)
        baseline_mse = float(np.mean((baseline_values - returns_np) ** 2))

        # 4. Policy gradient с advantages: (1-alpha)*policy_loss + alpha*entropy_loss
        policy_loss, mean_entropy = self._compute_policy_loss_and_entropy(advantages_t)
        alpha = self.entropy_coef
        entropy_loss = -mean_entropy
        total_loss = (1.0 - alpha) * policy_loss + alpha * entropy_loss

        self.policy_optimizer.zero_grad()
        total_loss.backward()
        if getattr(self, "max_grad_norm", None) is not None:
            torch.nn.utils.clip_grad_norm_(
                self.policy_network.parameters(),
                self.max_grad_norm,
            )
        self.policy_optimizer.step()

        mean_return = float(returns[0])
        self.reset_trajectory()
        return {
            "policy_loss": policy_loss.item(),
            "baseline_mse": baseline_mse,
            "mean_return": mean_return,
            "entropy": mean_entropy.item(),
        }
