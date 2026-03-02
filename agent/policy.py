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
            logits = torch.where(mask, logits, torch.tensor(float("-inf"), device=logits.device))
        return logits

    def get_log_probs(self, state, action_indices, valid_actions_mask=None):
        """
        Получить log π(a|s) для выбранных действий.

        Input:
            state — наблюдение, shape (batch, observation_dim)
            action_indices — индексы выбранных действий, shape (batch,)
            valid_actions_mask — опционально, маска допустимых действий
        Output: log_probs — логарифмы вероятностей, shape (batch,)
        """
        logits = self.forward(state, valid_actions_mask)
        log_probs = F.log_softmax(logits, dim=-1)
        action_indices = _to_tensor(action_indices, device=logits.device, dtype=torch.long)
        if action_indices.dim() == 2:
            action_indices = action_indices.squeeze(-1)
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


class ValueNetwork(nn.Module):
    """
    Сеть оценки V(s). Вход — state, выход — один скаляр (ожидаемый return).
    """

    def __init__(self, observation_dim: int, hidden_dims: list):
        super().__init__()
        dims = [observation_dim] + list(hidden_dims) + [1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, state):
        """state -> V(s), shape (batch,) или (batch, 1)."""
        state = _to_tensor(state, device=next(self.parameters()).device)
        out = self.mlp(state)
        return out.squeeze(-1)
