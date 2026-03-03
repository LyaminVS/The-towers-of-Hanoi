"""
REINFORCE с baseline — уменьшение дисперсии через вычитание V(s).

Baseline: табличная оценка V(s) по истории последних N траекторий.
Таблица проиндексирована состояниями: V[state_to_index(s)] = V̂(s).

Оценка V(s) перед каждым градиентным шагом:
    1. V(s) = 0, c(s) = 0 для всех s
    2. for trajectory in trajectory_history:
           for (s, G) in trajectory:
               V(s) += (G - V(s)) / (c(s) + 1)
               c(s) += 1
"""

import os
from collections import deque

import torch
import torch.nn.functional as F
import numpy as np

from .base_agent import BaseAgent
from .policy import PolicyNetwork
from env.actions import action_to_index, index_to_action
from env.state import observation_to_state


def _valid_actions_mask(valid_actions: list, action_space: list, device) -> torch.Tensor:
    """Булева маска допустимых действий."""
    mask = torch.zeros(len(action_space), dtype=torch.bool, device=device)
    for a in valid_actions:
        mask[action_space.index(a)] = True
    return mask


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


class REINFORCEBaselineAgent(BaseAgent):
    """
    REINFORCE + табличный baseline по истории траекторий:
        ∇J(θ) = E[Σ_t (G_t - V(s_t)) ∇ log π(a|s)]

    V(s) — таблица размером num_sticks^num_disks.
    Перед каждым градиентным шагом пересчитывается по последним
    history_len эпизодам инкрементальным средним.
    """

    def __init__(self, observation_dim: int, action_space: list, config: dict):
        super().__init__(observation_dim, action_space, config)
        self.action_space = action_space
        action_dim = len(action_space)
        hidden_dims = config.get("hidden_dims", [64, 64])
        self.policy_network = PolicyNetwork(observation_dim, action_dim, hidden_dims)
        lr = config.get("learning_rate", 1e-3)
        self.policy_optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=lr)
        self.gamma = config.get("discount_factor", 0.99)

        self.num_disks = observation_dim // 2
        self.num_sticks = config.get("num_sticks", 3)
        history_len = config.get("history_len", 20)

        # Кольцевой буфер последних history_len траекторий.
        # Каждая траектория — list[(state_tuple, G_t)].
        self.trajectory_history: deque = deque(maxlen=history_len)

        self.saved_valid_actions: list = []
        self._last_valid_actions: list = []

    # ------------------------------------------------------------------
    # Trajectory management
    # ------------------------------------------------------------------

    def store_transition(self, state, action, reward: float, log_prob: float) -> None:
        super().store_transition(state, action, reward, log_prob)
        self.saved_valid_actions.append(self._last_valid_actions)

    def reset_trajectory(self) -> None:
        super().reset_trajectory()
        self.saved_valid_actions.clear()

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, state, valid_actions: list, training: bool = True):
        self._last_valid_actions = list(valid_actions)
        device = next(self.policy_network.parameters()).device
        mask = _valid_actions_mask(valid_actions, self.action_space, device)
        state_t = torch.as_tensor(np.asarray(state), dtype=torch.float32, device=device)
        if state_t.dim() == 1:
            state_t = state_t.unsqueeze(0)
        mask_batch = mask.unsqueeze(0)

        with torch.no_grad():
            logits = self.policy_network.forward(state_t, mask_batch)
            probs = F.softmax(logits, dim=-1)
            if training:
                dist = torch.distributions.Categorical(probs=probs)
                action_idx = dist.sample().item()
            else:
                action_idx = probs.argmax(dim=-1).item()

        action = index_to_action(action_idx, self.action_space)
        log_prob = self.policy_network.get_log_probs(
            state_t, [action_idx], mask_batch
        ).item()
        return action, log_prob

    # ------------------------------------------------------------------
    # Returns & baseline
    # ------------------------------------------------------------------

    def _compute_returns(self) -> np.ndarray:
        rewards = self.saved_rewards
        T = len(rewards)
        returns = np.zeros(T, dtype=np.float64)
        G = 0.0
        for t in reversed(range(T)):
            G = rewards[t] + self.gamma * G
            returns[t] = G
        return returns

    def _predict_baseline(self, observations: list, V: np.ndarray) -> np.ndarray:
        """V(s) для каждого наблюдения из сохранённой траектории."""
        values = np.zeros(len(observations), dtype=np.float64)
        for i, obs in enumerate(observations):
            state = observation_to_state(obs)
            values[i] = V[state_to_index(state, self.num_sticks)]
        return values

    # ------------------------------------------------------------------
    # Value estimation from trajectory history
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Policy update
    # ------------------------------------------------------------------

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
        advantages = returns - baseline_values
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=device)
        baseline_mse = float(np.mean((baseline_values - returns) ** 2))

        # 4. Policy gradient: L(θ) = -(1/T) Σ_t A_t * log π(a_t | s_t)
        policy_loss = torch.tensor(0.0, device=device)
        n = len(self.saved_states)
        for t in range(n):
            state = self.saved_states[t]
            action = self.saved_actions[t]
            action_idx = action_to_index(action, self.action_space)
            valid_actions = (
                self.saved_valid_actions[t]
                if t < len(self.saved_valid_actions)
                else self.action_space
            )
            mask = _valid_actions_mask(valid_actions, self.action_space, device)
            state_t = torch.as_tensor(
                np.asarray(state), dtype=torch.float32, device=device
            ).unsqueeze(0)
            log_prob = self.policy_network.get_log_probs(
                state_t, [action_idx], mask.unsqueeze(0)
            ).squeeze(0)
            policy_loss = policy_loss - advantages_t[t] * log_prob

        policy_loss = policy_loss / max(n, 1)
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        mean_return = float(returns[0])
        self.reset_trajectory()
        return {
            "policy_loss": policy_loss.item(),
            "baseline_mse": baseline_mse,
            "mean_return": mean_return,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        save_dict = {
            "policy_network": self.policy_network.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
        }
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(save_dict, path)

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if "policy_network" in checkpoint:
            self.policy_network.load_state_dict(checkpoint["policy_network"])
        if "policy_optimizer" in checkpoint:
            self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
