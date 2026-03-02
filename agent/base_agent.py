"""
Базовый класс агента для policy gradient методов.

Единый интерфейс для REINFORCE, REINFORCE+baseline, TRPO и др.
"""

from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """
    Абстрактный базовый класс агента.
    
    Все policy gradient методы:
    — собирают траекторию (state, action, reward, log_prob) за эпизод
    — обновляют политику в конце эпизода по накопленной траектории
    """

    def __init__(self, observation_dim: int, action_space: list, config: dict):
        """
        Инициализация агента.
        
        Input:
            observation_dim — размерность наблюдения (state/observation)
            action_space — список всех действий [(from, to), ...] из get_action_space
            config — словарь гиперпараметров: learning_rate, discount_factor,
                    hidden_dims, и др. в зависимости от метода
        Output: —
        """
        ...

    @abstractmethod
    def select_action(self, state, valid_actions: list, training: bool = True):
        """
        Выбрать действие по текущей политике.
        
        Input:
            state — текущее наблюдение (формат env)
            valid_actions — список допустимых действий в данном состоянии
            training — если True, использовать exploration (например, сэмплирование);
                      если False, greedy (для оценки)
        Output: (action, log_prob)
            action — выбранное действие (from_stick, to_stick)
            log_prob — логарифм вероятности выбора этого действия (для update)
        """
        ...

    def store_transition(self, state, action, reward: float, log_prob: float) -> None:
        """
        Сохранить переход в буфер траектории (для policy gradient).
        
        Вызывается после каждого step в эпизоде. В конце эпизода update() использует
        накопленную траекторию.
        
        Input:
            state — состояние на момент выбора действия
            action — выполненное действие
            reward — полученная награда
            log_prob — log π(a|s) для этого действия
        Output: None
        """
        ...

    def reset_trajectory(self) -> None:
        """
        Очистить буфер траектории в начале нового эпизода.
        
        Input: —
        Output: None
        """
        ...

    @abstractmethod
    def update(self) -> dict:
        """
        Обновить политику по накопленной траектории. Вызывается в конце эпизода.
        
        Input: —
        Output: dict с метриками обновления (policy_loss, value_loss, kl, mean_return, ...)
        """
        ...

    def save(self, path: str) -> None:
        """
        Сохранить модель на диск (policy, value function, оптимизаторы).
        
        Input:
            path — путь к файлу (например "model.pth")
        Output: None
        """
        ...

    def load(self, path: str) -> None:
        """
        Загрузить модель с диска.
        
        Input:
            path — путь к файлу с сохранённой моделью
        Output: None
        Raises: FileNotFoundError если файл не найден
        """
        ...
