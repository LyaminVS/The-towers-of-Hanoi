# FILE: ./agent/__init__.py
from .reinforce import REINFORCEAgent
from .trpo import TRPOAgent
from .baseline import ZeroBaseline, TabularBaseline


def create_baseline(value_estimator: str, config: dict):
    """
    Фабрика для создания объекта baseline по имени метода.

    Input:
        value_estimator — "zero" | "tabular"
        config          — словарь с ключами num_disks, num_sticks, history_len
    Output: ZeroBaseline | TabularBaseline
    """
    if value_estimator == "zero":
        return ZeroBaseline()
    elif value_estimator == "tabular":
        return TabularBaseline(
            num_disks=config["num_disks"],
            num_sticks=config.get("num_sticks", 3),
            history_len=config.get("history_len", 20),
        )
    else:
        raise ValueError(f"Unknown value_estimator: '{value_estimator}'. Use 'zero' or 'tabular'.")


def create_agent(agent_method: str, observation_dim: int, action_space: list, config: dict, baseline):
    """
    Фабрика для создания агента по названию метода.

    Input:
        agent_method    — "reinforce" | "trpo"
        observation_dim — размерность наблюдения
        action_space    — список допустимых действий
        config          — словарь гиперпараметров
        baseline        — объект baseline (ZeroBaseline или TabularBaseline)
    Output: REINFORCEAgent | TRPOAgent
    """
    agents = {
        "reinforce": REINFORCEAgent,
        "trpo": TRPOAgent,
    }

    if agent_method not in agents:
        raise ValueError(f"Unknown agent method: '{agent_method}'. Use 'reinforce' or 'trpo'.")

    return agents[agent_method](observation_dim, action_space, config, baseline)
