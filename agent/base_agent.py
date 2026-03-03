"""
Базовый класс агента для policy gradient методов.

Единый интерфейс для REINFORCE, REINFORCE+baseline, TRPO и др.
"""

import os
from abc import ABC, abstractmethod
import torch


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
        self.observation_dim = observation_dim
        self.action_space = action_space
        self.config = config

        # Буферы для хранения траектории текущего эпизода
        self.saved_states = []
        self.saved_actions = []
        self.saved_rewards = []
        self.saved_log_probs = []

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
        pass

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
        self.saved_states.append(state)
        self.saved_actions.append(action)
        self.saved_rewards.append(reward)
        self.saved_log_probs.append(log_prob)

    def reset_trajectory(self) -> None:
        """
        Очистить буфер траектории в начале нового эпизода.
        
        Input: —
        Output: None
        """
        self.saved_states.clear()
        self.saved_actions.clear()
        self.saved_rewards.clear()
        self.saved_log_probs.clear()

    @abstractmethod
    def update(self) -> dict:
        """
        Обновить политику по накопленной траектории. Вызывается в конце эпизода.
        
        Input: —
        Output: dict с метриками обновления (policy_loss, value_loss, kl, mean_return, ...)
        """
        pass

    def save(self, path: str) -> None:
        """
        Сохранить модель на диск (policy, value function, оптимизаторы).
        
        Input:
            path — путь к файлу (например "model.pth")
        Output: None
        """
        save_dict = {}
        
        # Динамически сохраняем сети, если они созданы в дочернем классе
        if hasattr(self, 'policy_network') and self.policy_network is not None:
            save_dict['policy_network'] = self.policy_network.state_dict()
        if hasattr(self, 'value_network') and self.value_network is not None:
            save_dict['value_network'] = self.value_network.state_dict()
            
        # Сохраняем оптимизаторы
        if hasattr(self, 'optimizer') and self.optimizer is not None:
            save_dict['optimizer'] = self.optimizer.state_dict()
        if hasattr(self, 'policy_optimizer') and self.policy_optimizer is not None:
            save_dict['policy_optimizer'] = self.policy_optimizer.state_dict()
        if hasattr(self, 'value_optimizer') and self.value_optimizer is not None:
            save_dict['value_optimizer'] = self.value_optimizer.state_dict()

        # Создаем директорию, если её нет
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(save_dict, path)

    def load(self, path: str) -> None:
        """
        Загрузить модель с диска.
        
        Input:
            path — путь к файлу с сохранённой моделью
        Output: None
        Raises: FileNotFoundError если файл не найден
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at: {path}")

        checkpoint = torch.load(path, map_location=torch.device("cpu"), weights_only=False)

        if hasattr(self, "policy_network") and "policy_network" in checkpoint:
            self.policy_network.load_state_dict(checkpoint["policy_network"])
        if hasattr(self, "value_network") and "value_network" in checkpoint:
            self.value_network.load_state_dict(checkpoint["value_network"])

        # Восстанавливаем состояния оптимизаторов
        if hasattr(self, 'optimizer') and 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if hasattr(self, 'policy_optimizer') and 'policy_optimizer' in checkpoint:
            self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        if hasattr(self, 'value_optimizer') and 'value_optimizer' in checkpoint:
            self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])