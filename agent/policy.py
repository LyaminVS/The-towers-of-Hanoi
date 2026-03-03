import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def _to_tensor(x, device=None, dtype=torch.float32):
    """Convert array or tensor to torch tensor, ensure 2D (batch, dim)."""
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(np.asarray(x), dtype=dtype, device=device)
    else:
        x = x.to(dtype=dtype, device=device)
    if x.dim() == 1:
        x = x.unsqueeze(0)
    return x


class PolicyNetwork(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dims: list):
        """
        Инициализация сети.

        Input:
            observation_dim — размерность входа (state/observation)
            action_dim — размерность выхода (количество действий = len(action_space))
            hidden_dims — список размеров скрытых слоёв, например [64, 64]
        Output: —
        """
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim

        dims = [observation_dim] + list(hidden_dims) + [action_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, state, valid_actions_mask=None):
        """
        Прямой проход: state -> logits по всем действиям.

        Input:
            state — наблюдение (tensor или array), shape (batch, observation_dim)
            valid_actions_mask — опционально: булева маска допустимых действий,
                               shape (batch, action_dim); недопустимые = -inf для softmax
        Output: logits — логиты по всем действиям, shape (batch, action_dim)
        """
        state = _to_tensor(state, device=next(self.parameters()).device)
        logits = self.mlp(state)
        if valid_actions_mask is not None:
            mask = _to_tensor(valid_actions_mask, device=logits.device, dtype=torch.bool)
            logits = torch.where(mask, logits, torch.tensor(-1e9, dtype=logits.dtype, device=logits.device))
        return logits

    def get_log_probs(self, state, action_indices, valid_actions_mask=None):
        logits = self.forward(state, valid_actions_mask)
        log_probs = F.log_softmax(logits, dim=-1)

        # --- FIX: ensure batch dimension exists ---
        if log_probs.dim() == 1:          # [A] -> [1, A]
            log_probs = log_probs.unsqueeze(0)

        # Convert action_indices to tensor without auto-expanding to 2D
        if not isinstance(action_indices, torch.Tensor):
            action_indices = torch.as_tensor(action_indices, dtype=torch.long, device=log_probs.device)
        else:
            action_indices = action_indices.to(dtype=torch.long, device=log_probs.device)

        # action_indices can be scalar, 1D [B], or 2D [B,1] depending on caller
        if action_indices.dim() == 0:      # [] -> [1]
            action_indices = action_indices.unsqueeze(0)
        elif action_indices.dim() == 2 and action_indices.size(-1) == 1:  # [B,1] -> [B]
            action_indices = action_indices.squeeze(-1)

        # Now: log_probs [B, A], action_indices [B]
        return log_probs.gather(dim=-1, index=action_indices.unsqueeze(-1)).squeeze(-1)


    def get_entropy(self, state, valid_actions_mask=None):
        """
        Энтропия распределения π(·|s).

        Input:
            state — наблюдение
            valid_actions_mask — опционально
        Output: entropy — скаляр или tensor (batch,)
        """
        logits = self.forward(state, valid_actions_mask)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1)
        return entropy


import torch
import torch.nn as nn

class ValueNetwork(nn.Module):
    """
    Линейный критик: V(s) = w^T s + b
    Эквивалент линейной регрессии (по сути, baseline как линейная модель).
    """

    def __init__(self, observation_dim: int, hidden_dims: list):
        super().__init__()
        self.observation_dim = observation_dim

        # Единственный линейный слой -> скаляр V(s)
        self.linear = nn.Linear(observation_dim, 1)

        # hidden_dims оставляем ради совместимости, но не используем

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        state: Tensor shape [B, observation_dim] или [observation_dim]
        returns: Tensor shape [B] (скаляр на каждый state)
        """
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32, device=next(self.parameters()).device)
        else:
            state = state.to(dtype=torch.float32, device=next(self.parameters()).device)

        if state.dim() == 1:
            state = state.unsqueeze(0)

        v = self.linear(state)          # [B, 1]
        return v.squeeze(-1)            # [B]