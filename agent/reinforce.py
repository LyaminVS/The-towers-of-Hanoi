"""
REINFORCE — vanilla policy gradient (монте-карло оценка градиента).
"""

from .base_agent import BaseAgent


class REINFORCEAgent(BaseAgent):
    """
    REINFORCE: ∇J(θ) = E[Σ G_t ∇ log π(a|s)].
    Использует полный return G_t до конца эпизода (монте-карло).
    """

    def __init__(self, observation_dim: int, action_space: list, config: dict):
        """
        Input:
            observation_dim — размерность state
            action_space — список действий из get_action_space
            config — learning_rate, discount_factor (gamma), hidden_dims
        Output: —
        """
        ...

    def select_action(self, state, valid_actions: list, training: bool = True):
        """
        Сэмплировать действие из π(a|s). Сохранить log_prob для update.
        
        Input: state, valid_actions, training (если False — greedy)
        Output: (action, log_prob)
        """
        ...

    def _compute_returns(self) -> list:
        """
        Вычислить discounted returns G_t = Σ γ^k * r_{t+k} для каждого шага.
        
        Input: —
        Output: список [G_0, G_1, ...] — return с шага t до конца эпизода
        """
        ...

    def update(self) -> dict:
        """
        REINFORCE update: градиент policy loss = -Σ G_t * log π(a_t|s_t).
        
        Input: —
        Output: dict с policy_loss, mean_return
        """
        ...
