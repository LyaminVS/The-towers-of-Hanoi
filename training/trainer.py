"""
Цикл обучения RL-агента на среде Tower of Hanoi.

Для policy gradient (REINFORCE, TRPO): в каждом шаге — agent.store_transition(),
в конце эпизода — agent.update().
"""


def run_episode(
    env,
    agent,
    max_steps: int,
    use_death_penalty: bool = False,
) -> tuple[float, int, bool]:
    """
    Запустить один эпизод обучения.
    
    Цикл: reset → loop(select_action, env.step, store_transition) до done/truncated.
    В конце эпизода вызывается agent.update().
    
    Input:
        env — среда TowerOfHanoiEnv
        agent — RL-агент (BaseAgent)
        max_steps — максимальное число шагов (для truncated)
        use_death_penalty — использовать ли reward.use_death_penalty (эпизод
                           завершается при недопустимом ходе)
    Output: (total_reward, num_steps, success)
        total_reward — сумма наград за эпизод
        num_steps — количество выполненных шагов
        success — True если цель достигнута (все диски на 3-й палке)
    """
    ...


def train(
    env,
    agent,
    num_episodes: int,
    max_steps_per_episode: int,
    use_death_penalty: bool = False,
    log_interval: int = 100,
) -> list[dict]:
    """
    Полный цикл обучения: num_episodes эпизодов.
    
    Input:
        env — среда
        agent — RL-агент
        num_episodes — количество эпизодов
        max_steps_per_episode — лимит шагов на эпизод
        use_death_penalty — использовать ли штраф смерти
        log_interval — интервал логирования (каждые N эпизодов)
    Output: список словарей с метриками по эпизодам
            [{reward, steps, success, **update_metrics}, ...]
    """
    ...
