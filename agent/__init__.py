"""RL-агенты (policy gradient методы)."""
from .base_agent import BaseAgent
from .policy import PolicyNetwork, ValueNetwork
from .reinforce import REINFORCEAgent
from .reinforce_baseline import REINFORCEBaselineAgent
from .trpo import TRPOAgent

AGENT_METHODS = ("reinforce", "reinforce_baseline", "trpo")


def create_agent(method: str, observation_dim: int, action_space: list, config: dict) -> BaseAgent:
    """
    Фабрика: создать агента по имени метода.
    
    Input:
        method — "reinforce" | "reinforce_baseline" | "trpo"
        observation_dim — размерность наблюдения
        action_space — список действий из env.get_action_space()
        config — словарь гиперпараметров (из config.settings или переопределённые)
    Output: экземпляр BaseAgent (REINFORCEAgent, REINFORCEBaselineAgent или TRPOAgent)
    Raises: ValueError если method не в AGENT_METHODS
    """
    ...
