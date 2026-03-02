"""
REINFORCE с baseline — уменьшение дисперсии через вычитание V(s).
"""

from .base_agent import BaseAgent


class REINFORCEBaselineAgent(BaseAgent):
    """
    REINFORCE + baseline: ∇J(θ) = E[Σ (G_t - V(s_t)) ∇ log π(a|s)].
    Advantage A_t = G_t - V(s_t) снижает дисперсию градиента.
    """

    def __init__(self, observation_dim: int, action_space: list, config: dict):
        """
        Input:
            observation_dim, action_space
            config — learning_rate, value_lr (для value network), discount_factor, hidden_dims
        Output: —
        """
        ...

    def select_action(self, state, valid_actions: list, training: bool = True):
        """
        Сэмплировать действие из π(a|s). Сохранить log_prob и state для update.
        
        Input: state, valid_actions, training
        Output: (action, log_prob)
        """
        ...

    def _compute_returns(self) -> list:
        """
        Вычислить discounted returns G_t.
        
        Input: —
        Output: список returns
        """
        ...

    def _compute_baseline(self, states) -> list:
        """
        Получить V(s) для каждого состояния в траектории.
        
        Input: states — список состояний из буфера траектории
        Output: список значений baseline (скаляры)
        """
        ...

    def update(self) -> dict:
        """
        Update policy: advantage (G_t - V(s_t)) * ∇ log π.
        Update value: минимизация MSE(V(s_t), G_t).
        
        Input: —
        Output: dict с policy_loss, value_loss, mean_return
        """
        ...
