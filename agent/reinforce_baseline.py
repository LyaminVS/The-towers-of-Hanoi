"""
REINFORCE с baseline — уменьшение дисперсии через вычитание V(s).
"""

import torch
import torch.nn.functional as F
import numpy as np

from .base_agent import BaseAgent
from .policy import PolicyNetwork, ValueNetwork
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
    REINFORCE + baseline: ∇J(θ) = E[Σ (G_t - V(s_t)) ∇ log π(a|s)].
    Advantage A_t = G_t - V(s_t) снижает дисперсию градиента.
    """

    def __init__(self, observation_dim: int, action_space: list, config: dict):
        super().__init__(observation_dim, action_space, config)
        self.action_space = action_space
        action_dim = len(action_space)
        hidden_dims = config.get("hidden_dims", [64, 64])
        value_ridge = float(config.get("value_ridge", 1e-3))
        self.policy_network = PolicyNetwork(observation_dim, action_dim, hidden_dims)
        self.value_network = ValueNetwork(observation_dim, hidden_dims, value_ridge=value_ridge)
        lr = config.get("learning_rate", 1e-3)
        value_lr = config.get("value_lr", 1e-3)
        self.policy_optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=lr)
        self.value_optimizer = torch.optim.Adam(self.value_network.parameters(), lr=value_lr)
        self.optimizer = None
        self.gamma = config.get("discount_factor", 0.99)
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

    def _compute_returns(self) -> list:
        rewards = self.saved_rewards
        T = len(rewards)
        returns = []
        G = 0.0
        for t in reversed(range(T)):
            G = rewards[t] + self.gamma * G
            returns.append(G)
        return list(reversed(returns))

    def _compute_baseline(self, states) -> list:
        """V(s) для каждого состояния в траектории."""
        if not states:
            return []
        device = next(self.value_network.parameters()).device
        batch = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=device)
        with torch.no_grad():
            values = self.value_network(batch)
        return values.cpu().tolist() if values.dim() > 0 else [values.item()]

    def update(self) -> dict:
        if not self.saved_states:
            return {"policy_loss": 0.0, "value_loss": 0.0, "mean_return": 0.0}

        device = next(self.policy_network.parameters()).device
        returns = self._compute_returns()
        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)

        states_batch = torch.as_tensor(
            np.asarray(self.saved_states), dtype=torch.float32, device=device
        )
        values_t = self.value_network(states_batch)

        advantages = returns_t - values_t.detach()

        policy_loss = 0.0
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
            mask_batch = mask.unsqueeze(0)
            log_prob = self.policy_network.get_log_probs(
                state_t, [action_idx], mask_batch
            ).squeeze(0)
            policy_loss = policy_loss - (advantages[t] * log_prob).sum()
        policy_loss = policy_loss / max(n, 1)

        value_loss = F.mse_loss(values_t, returns_t)

        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        (policy_loss + value_loss).backward()
        self.policy_optimizer.step()
        self.value_optimizer.step()

        mean_return = returns[0] if returns else 0.0
        self.reset_trajectory()
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "mean_return": mean_return,
        }
