"""
REINFORCE с baseline — уменьшение дисперсии через вычитание V̂(s).

Baseline: линейная регрессия V̂(s; w, b) = wᵀ s + b,
обновляемая по MSE-градиенту после каждого эпизода.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np

from .base_agent import BaseAgent
from .policy import PolicyNetwork
from env.actions import action_to_index, index_to_action


def _valid_actions_mask(valid_actions: list, action_space: list, device) -> torch.Tensor:
    """Булева маска допустимых действий: True для действий из valid_actions."""
    mask = torch.zeros(len(action_space), dtype=torch.bool, device=device)
    for a in valid_actions:
        idx = action_space.index(a)
        mask[idx] = True
    return mask


class REINFORCEBaselineAgent(BaseAgent):
    """
    REINFORCE + линейный baseline:
        ∇J(θ) = E[Σ_t (G_t - V̂(s_t)) ∇ log π(a|s)]

    V̂(s; w, b) = wᵀ s + b — линейная регрессия без нейросети.
    Веса w и b обновляются по градиенту MSE после каждого эпизода:
        L = (1/T) Σ_t (V̂(s_t) - G_t)²
    """

    def __init__(self, observation_dim: int, action_space: list, config: dict):
        super().__init__(observation_dim, action_space, config)
        self.action_space = action_space
        action_dim = len(action_space)
        hidden_dims = config.get("hidden_dims", [64, 64])
        self.policy_network = PolicyNetwork(observation_dim, action_dim, hidden_dims)
        lr = config.get("learning_rate", 1e-3)
        self.policy_optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=lr)
        self.optimizer = None
        self.gamma = config.get("discount_factor", 0.99)
        self.baseline_lr = config.get("baseline_lr", 1e-2)

        # Линейный бейзлайн: V̂(s) = wᵀ s + bias
        self.baseline_weights = np.zeros(observation_dim + 1, dtype=np.float64)

        self.saved_valid_actions: list = []
        self._last_valid_actions: list = []

    def store_transition(self, state, action, reward: float, log_prob: float) -> None:
        # возможно в базовом классе это стоит добавить
        super().store_transition(state, action, reward, log_prob)
        self.saved_valid_actions.append(self._last_valid_actions)

    def reset_trajectory(self) -> None:
        # возможно в базовом классе это стоит добавить
        super().reset_trajectory()
        self.saved_valid_actions.clear()

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
        log_prob = self.policy_network.get_log_probs(state_t, [action_idx], mask_batch).item()
        return action, log_prob

    def _compute_returns(self) -> np.ndarray:
        rewards = self.saved_rewards
        T = len(rewards)
        returns = np.zeros(T, dtype=np.float64)
        G = 0.0
        for t in reversed(range(T)):
            G = rewards[t] + self.gamma * G
            returns[t] = G
        return returns

    def _predict_baseline(self, states_np: np.ndarray) -> np.ndarray:
        """V̂(s) = wᵀ s + bias для батча состояний shape (T, obs_dim)."""
        return states_np @ self.baseline_weights[:-1] + self.baseline_weights[-1]

    def _update_baseline(self, states_np: np.ndarray, returns: np.ndarray) -> float:
        """
        Один градиентный шаг по MSE: L = (1/T) Σ (V̂(s_t) - G_t)².
        Обновляет baseline_weights и baseline_bias.
        Возвращает MSE до обновления (для логирования).
        """
        T = len(returns)
        residuals = self._predict_baseline(states_np) - returns  # (T,)
        mse = float(np.mean(residuals ** 2))
        X = np.concatenate([states_np, np.ones((states_np.shape[0], 1))], axis=1)
        self.baseline_weights = np.linalg.pinv(X.T @ X) @ X.T @ returns
        return mse

    def update(self) -> dict:
        if not self.saved_states:
            return {"policy_loss": 0.0, "baseline_mse": 0.0, "mean_return": 0.0}

        device = next(self.policy_network.parameters()).device
        returns = self._compute_returns()
        states_np = np.asarray(self.saved_states, dtype=np.float64)

        # Advantages вычисляем с текущим (ещё не обновлённым) бейзлайном
        advantages = returns - self._predict_baseline(states_np)
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=device)

        # Обновляем линейный бейзлайн по MSE
        baseline_mse = self._update_baseline(states_np, returns)

        # Policy gradient: L(θ) = -(1/T) Σ_t A_t * log π(a_t | s_t)
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

    def save(self, path: str) -> None:
        save_dict = {
            "policy_network": self.policy_network.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "baseline_weights": self.baseline_weights,
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
        if "baseline_weights" in checkpoint:
            self.baseline_weights = checkpoint["baseline_weights"]
