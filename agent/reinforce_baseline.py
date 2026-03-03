"""
REINFORCE с baseline — уменьшение дисперсии через вычитание V(s).

Baseline: табличная оценка V(s), обновляемая методом Монте-Карло.
Таблица проиндексирована состояниями: V[state_to_index(s)] = V̂(s).

MC-обновление после каждого эпизода (до обновления политики):
    - V(s) = 0 для всех s в начале каждой итерации (сбрасывается)
    - K роллаутов текущей политики (training=True, без градиентов)
    - Инкрементальное среднее: V(s) += (G_t - V(s)) / visit_count(s)
"""

import os
import torch
import torch.nn.functional as F
import numpy as np

from .base_agent import BaseAgent
from .policy import PolicyNetwork
from env.actions import action_to_index, index_to_action, get_valid_actions
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
    REINFORCE + табличный MC-baseline:
        ∇J(θ) = E[Σ_t (G_t - V(s_t)) ∇ log π(a|s)]

    V(s) — таблица размером num_sticks^num_disks.
    Обновляется каждый эпизод: K MC-роллаутов → инкрементальное среднее.
    Требует self.env — ссылку на среду, устанавливаемую через trainer.
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
        self.mc_episodes = config.get("mc_episodes", 5)

        # Ссылка на среду для MC-роллаутов. Устанавливается из trainer.py.
        self.env = None

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
    # Monte Carlo estimation of V(s)
    # ------------------------------------------------------------------

    def _run_mc_rollout(self) -> list:
        """
        Один роллаут текущей политики без обновления весов.
        Возвращает список (state_tuple, G_t) — пары (состояние, дисконтированный доход).
        """
        obs, info = self.env.reset(random_init=True)
        trajectory = []  # [(state_tuple, reward), ...]

        while True:
            state_tuple = tuple(tuple(d) for d in info["state"])
            valid_actions = get_valid_actions(state_tuple, self.env.num_sticks)
            action, _ = self.select_action(obs, valid_actions, training=True)
            obs, reward, terminated, truncated, info = self.env.step(action)
            trajectory.append((state_tuple, reward))
            if terminated or truncated:
                break

        # Дисконтированные доходы G_t = r_t + γ r_{t+1} + ...
        T = len(trajectory)
        returns = [0.0] * T
        G = 0.0
        for t in reversed(range(T)):
            G = trajectory[t][1] + self.gamma * G
            returns[t] = G

        return [(trajectory[t][0], returns[t]) for t in range(T)]

    def _update_value_table(self) -> np.ndarray:
        """
        MC-оценка V(s): mc_episodes роллаутов → every-visit инкрементальное среднее.

        V(s) = 0 в начале каждой итерации (сбрасывается, не накапливается).
        Возвращает таблицу V размером num_sticks^num_disks.
        """
        num_states = self.num_sticks ** self.num_disks
        V = np.zeros(num_states, dtype=np.float64)
        n = np.zeros(num_states, dtype=np.int64)

        if self.env is None:
            return V

        for _ in range(self.mc_episodes):
            rollout = self._run_mc_rollout()
            for state_tuple, G_t in rollout:
                idx = state_to_index(state_tuple, self.num_sticks)
                n[idx] += 1
                V[idx] += (G_t - V[idx]) / n[idx]

        return V

    # ------------------------------------------------------------------
    # Policy update
    # ------------------------------------------------------------------

    def update(self) -> dict:
        if not self.saved_states:
            return {"policy_loss": 0.0, "baseline_mse": 0.0, "mean_return": 0.0}

        device = next(self.policy_network.parameters()).device
        returns = self._compute_returns()

        # 1. MC-оценка V(s) до обновления политики (V сбрасывается каждую итерацию)
        V = self._update_value_table()

        # 2. Advantages = G_t - V(s_t)
        baseline_values = self._predict_baseline(self.saved_states, V)
        advantages = returns - baseline_values
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=device)
        baseline_mse = float(np.mean((baseline_values - returns) ** 2))

        # 3. Policy gradient: L(θ) = -(1/T) Σ_t A_t * log π(a_t | s_t)
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
