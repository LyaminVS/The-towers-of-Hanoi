# FILE: ./agent/__init__.py
from .reinforce import REINFORCEAgent
from .reinforce_baseline import REINFORCEBaselineAgent
from .trpo import TRPOAgent

def create_agent(agent_method, observation_dim, action_space, config):
    """Фабрика для создания агента по названию метода."""
    agents = {
        "reinforce": REINFORCEAgent,
        "reinforce_baseline": REINFORCEBaselineAgent,
        "trpo": TRPOAgent
    }
    
    if agent_method not in agents:
        raise ValueError(f"Unknown agent method: {agent_method}")
        
    return agents[agent_method](observation_dim, action_space, config)