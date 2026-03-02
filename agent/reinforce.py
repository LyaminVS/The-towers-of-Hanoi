"""
REINFORCE — vanilla policy gradient (монте-карло оценка градиента).
"""

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


class REINFORCEAgent(BaseAgent):
    """
    REINFORCE: ∇J(θ) = E[Σ G_t ∇ log π(a|s)].
    Использует полный return G_t до конца эпизода (монте-карло).
    """

    def __init__(self, observation_dim: int, action_space: list, config: dict):
        super().__init__(observation_dim, action_space, config)
        self.action_space = action_space
        action_dim = len(action_space)
        hidden_dims = config.get("hidden_dims", [64, 64])
        self.policy_network = PolicyNetwork(observation_dim, action_dim, hidden_dims)
        self.optimizer = None
        self.policy_optimizer = torch.optim.Adam(
            self.policy_network.parameters(),
            lr=config.get("learning_rate", 1e-3),
        )
        self.gamma = config.get("discount_factor", 0.99)
        self.value_network = None
        self.saved_valid_actions: list = []
        self._last_valid_actions: list = []
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

    def update(self) -> dict:
        if not self.saved_states:
            return {"policy_loss": 0.0, "mean_return": 0.0}

        device = next(self.policy_network.parameters()).device
        returns = self._compute_returns()
        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)

        policy_loss = 0.0
        n = len(self.saved_states)
        for t in range(n):
            state = self.saved_states[t]
            action = self.saved_actions[t]
            action_idx = action_to_index(action, self.action_space)
            valid_actions = self.saved_valid_actions[t] if t < len(self.saved_valid_actions) else self.action_space
            mask = _valid_actions_mask(valid_actions, self.action_space, device)
            state_t = torch.as_tensor(np.asarray(state), dtype=torch.float32, device=device).unsqueeze(0)
            mask_batch = mask.unsqueeze(0)
            log_prob = self.policy_network.get_log_probs(state_t, [action_idx], mask_batch).squeeze(0)
            policy_loss = policy_loss - (returns_t[t] * log_prob).sum()
        policy_loss = policy_loss / max(n, 1)

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        mean_return = returns[0] if returns else 0.0
        self.reset_trajectory()
        return {"policy_loss": policy_loss.item(), "mean_return": mean_return}
