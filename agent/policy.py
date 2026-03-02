"""
Нейронные сети для policy gradient: policy и value.
"""


class PolicyNetwork:
    """
    Сеть политики π(a|s). Вход — state, выход — logits по действиям.
    Для дискретных действий с маскировкой недопустимых (valid_actions_mask).
    """

    def __init__(self, observation_dim: int, action_dim: int, hidden_dims: list):
        """
        Инициализация сети.
        
        Input:
            observation_dim — размерность входа (state/observation)
            action_dim — размерность выхода (количество действий = len(action_space))
            hidden_dims — список размеров скрытых слоёв, например [64, 64]
        Output: —
        """
        ...

    def forward(self, state, valid_actions_mask=None):
        """
        Прямой проход: state → logits по всем действиям.
        
        Input:
            state — наблюдение (tensor или array), shape (batch, observation_dim)
            valid_actions_mask — опционально: булева маска допустимых действий,
                               shape (batch, action_dim); недопустимые = -inf для softmax
        Output: logits — логиты по всем действиям, shape (batch, action_dim)
        """
        ...

    def get_log_probs(self, state, action_indices, valid_actions_mask=None):
        """
        Получить log π(a|s) для выбранных действий.
        
        Input:
            state — наблюдение, shape (batch, observation_dim)
            action_indices — индексы выбранных действий, shape (batch,)
            valid_actions_mask — опционально, маска допустимых действий
        Output: log_probs — логарифмы вероятностей, shape (batch,)
        """
        ...

    def get_entropy(self, state, valid_actions_mask=None):
        """
        Энтропия распределения π(·|s).
        
        Input:
            state — наблюдение
            valid_actions_mask — опционально
        Output: entropy — скаляр или tensor (batch,)
        """
        ...
