"""
REINFORCE — vanilla policy gradient с опциональным baseline.

Объект baseline передаётся снаружи (из run/*.py):
    ZeroBaseline    → vanilla REINFORCE (advantages = returns)
    TabularBaseline → REINFORCE + baseline (advantages = returns - V(s))

∇J(θ) = E[Σ_t (G_t - b(s_t)) ∇ log π(a_t | s_t)]
"""

import torch
import torch.nn.functional as F
import numpy as np

from .base_agent import BaseAgent
from .policy import PolicyNetwork
from env.actions import action_to_index, index_to_action
from env.state import observation_to_state


def _valid_actions_mask(valid_actions: list, action_space: list, device) -> torch.Tensor:
    """Булева маска допустимых действий: True для действий из valid_actions."""
    mask = torch.zeros(len(action_space), dtype=torch.bool, device=device)
    for a in valid_actions:
        mask[action_space.index(a)] = True
    return mask


class REINFORCEAgent(BaseAgent):
    """
    REINFORCE с передаваемым снаружи baseline.

    Config keys:
        learning_rate   (float, default 1e-3)
        discount_factor (float, default 0.99)
        hidden_dims     (list,  default [64, 64])
        entropy_coef    (float, default 0.01)
    """

    def __init__(self, observation_dim: int, action_space: list, config: dict, baseline):
        super().__init__(observation_dim, action_space, config)
        self.action_space = action_space
        self.baseline = baseline

        action_dim = len(action_space)
        hidden_dims = config.get("hidden_dims", [64, 64])
        self.policy_network = PolicyNetwork(observation_dim, action_dim, hidden_dims)
        self.optimizer = None
        self.policy_optimizer = torch.optim.Adam(
            self.policy_network.parameters(),
            lr=config.get("learning_rate", 1e-3),
        )
        self.gamma = config.get("discount_factor", 0.99)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.value_network = None
        self.saved_valid_actions: list = []
        self._last_valid_actions: list = []

    def store_transition(self, state, action, reward: float, log_prob: float) -> None:
        super().store_transition(state, action, reward, log_prob)
        self.saved_valid_actions.append(self._last_valid_actions)

    def reset_trajectory(self) -> None:
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
        T = len(self.saved_rewards)
        returns = np.zeros(T, dtype=np.float64)
        G = 0.0
        for t in reversed(range(T)):
            G = self.saved_rewards[t] + self.gamma * G
            returns[t] = G
        return returns

    def update(self) -> dict:
        if not self.saved_states:
            return {"policy_loss": 0.0, "mean_return": 0.0}

        device = next(self.policy_network.parameters()).device
        returns = self._compute_returns()
        n = len(self.saved_states)

        # Добавляем текущий эпизод в baseline и получаем b(s_t)
        episode_traj = [
            (observation_to_state(self.saved_states[t]), float(returns[t]))
            for t in range(n)
        ]
        baseline_values = self.baseline.predict(self.saved_states)
        self.baseline.add_trajectory(episode_traj)
        # baseline_values = self.baseline.predict(self.saved_states)

        # advantages = G_t - b(s_t)  [для ZeroBaseline: G_t - 0 = G_t]
        advantages = returns - baseline_values
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=device)
        baseline_mse = float(np.mean((baseline_values - returns) ** 2))

        policy_loss = torch.tensor(0.0, device=device)
        entropy_sum = 0.0

        for t in range(n):
            state = self.saved_states[t]
            action = self.saved_actions[t]
            action_idx = action_to_index(action, self.action_space)
            valid_actions = self.saved_valid_actions[t] if t < len(self.saved_valid_actions) else self.action_space
            mask = _valid_actions_mask(valid_actions, self.action_space, device)
            state_t = torch.as_tensor(np.asarray(state), dtype=torch.float32, device=device).unsqueeze(0)
            mask_batch = mask.unsqueeze(0)
            log_prob = self.policy_network.get_log_probs(state_t, [action_idx], mask_batch).squeeze(0)
            policy_loss = policy_loss - advantages_t[t] * log_prob
            entropy_sum = entropy_sum + self.policy_network.get_entropy(state_t, mask_batch).sum()

        policy_loss = policy_loss / max(n, 1)
        mean_entropy = entropy_sum / max(n, 1)
        total_loss = policy_loss - self.entropy_coef * mean_entropy

        self.policy_optimizer.zero_grad()
        total_loss.backward()
        self.policy_optimizer.step()

        self.reset_trajectory()
        return {
            "policy_loss": policy_loss.item(),
            "entropy": mean_entropy.item(),
            "baseline_mse": baseline_mse,
            "mean_return": float(returns[0]),
        }
