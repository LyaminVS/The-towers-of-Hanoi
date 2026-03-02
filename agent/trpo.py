"""
TRPO — Trust Region Policy Optimization.
Ограничение KL-дивергенции при обновлении политики.
"""

from .base_agent import BaseAgent


class TRPOAgent(BaseAgent):
    """
    TRPO: максимизация surrogate objective L с ограничением KL(π_old, π_new) ≤ δ.
    Использует conjugate gradient для вычисления направления и line search для шага.
    """

    def __init__(self, observation_dim: int, action_space: list, config: dict):
        """
        Input:
            observation_dim, action_space
            config — discount_factor, max_kl (δ), cg_iters (итерации conjugate gradient),
                    backtrack_iters, backtrack_coef (для line search), hidden_dims
        Output: —
        """
        ...

    def select_action(self, state, valid_actions: list, training: bool = True):
        """
        Сэмплировать действие из π(a|s). Сохранить log_prob, state для update.
        
        Input: state, valid_actions, training
        Output: (action, log_prob)
        """
        ...

    def _compute_returns_and_advantages(self) -> tuple:
        """
        Вычислить returns G_t и advantages A_t = G_t - V(s_t).
        
        Input: —
        Output: (returns, advantages) — списки или tensors
        """
        ...

    def _compute_kl(self, old_log_probs, new_log_probs) -> float:
        """
        Вычислить KL(π_old || π_new) = E[log π_old - log π_new] по траектории.
        
        Input:
            old_log_probs — log π_old(a|s) из траектории
            new_log_probs — log π_new(a|s) при текущих параметрах
        Output: mean KL (скаляр)
        """
        ...

    def _policy_gradient(self) -> "tensor":
        """
        Вычислить градиент surrogate objective L = Σ A_t * (π_new/π_old).
        
        Input: —
        Output: градиент (tensor)
        """
        ...

    def _fisher_vector_product(self, v: "tensor") -> "tensor":
        """
        Вычислить F*v, где F — Fisher information matrix.
        Используется в conjugate gradient для решения F*x = g.
        
        Input: v — вектор (tensor)
        Output: F*v (tensor)
        """
        ...

    def _line_search(self, step_dir, max_kl: float) -> float:
        """
        Найти коэффициент α такой, что KL ≤ max_kl и objective улучшается.
        Backtracking: α = 1, α*coef, α*coef^2, ...
        
        Input:
            step_dir — направление обновления (из conjugate gradient)
            max_kl — максимально допустимая KL-дивергенция
        Output: α — коэффициент шага (0 < α ≤ 1)
        """
        ...

    def update(self) -> dict:
        """
        TRPO update: conjugate gradient → direction → line search → apply.
        
        Input: —
        Output: dict с policy_loss, kl_divergence, improvement
        """
        ...
