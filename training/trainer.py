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
) -> tuple[float, int, bool, list[float], dict]:
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
    Output: (total_reward, num_steps, success, rewards, update_metrics)
        total_reward — сумма наград за эпизод
        num_steps — количество выполненных шагов
        success — True если цель достигнута (все диски на 3-й палке)
        rewards — список наград [r_0, r_1, ...] для расчёта дисконтированной награды
        update_metrics — dict из agent.update() (policy_loss, mean_return, ...)
    """
    env.reward.use_death_penalty = use_death_penalty
    env.max_steps = max_steps
    agent.reset_trajectory()

    observation, info = env.reset()
    total_reward = 0.0
    rewards: list[float] = []

    while True:
        state_before = observation
        action, log_prob = agent.select_action(observation, agent.action_space, training=True)

        observation, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        agent.store_transition(state_before, action, reward, log_prob)

        total_reward += reward
        num_steps = info["step_count"]
        done = terminated or truncated

        if done:
            break

    update_metrics = agent.update()
    success = info.get("is_correct_placement", 0) == env.num_disks
    return total_reward, num_steps, success, rewards, update_metrics


def _compute_discounted_return(rewards: list[float], gamma: float) -> float:
    """
    Вычислить дисконтированную награду G_0 = Σ γ^k * r_k.
    
    Input:
        rewards — список наград [r_0, r_1, ...]
        gamma — коэффициент дисконтирования
    Output: дисконтированная награда с начала эпизода
    """
    return sum(gamma**k * r for k, r in enumerate(rewards))


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
            [{reward, steps, success, discounted_return, **update_metrics}, ...]
    """
    gamma = agent.config.get("gamma", agent.config.get("discount_factor", 0.99))
    history: list[dict] = []

    for ep in range(num_episodes):
        total_reward, num_steps, success, rewards, update_metrics = run_episode(
            env, agent, max_steps_per_episode, use_death_penalty
        )
        discounted_return = _compute_discounted_return(rewards, gamma)

        episode_metrics = {
            "episode": ep + 1,
            "reward": total_reward,
            "steps": num_steps,
            "success": success,
            "discounted_return": discounted_return,
            **update_metrics,
        }
        history.append(episode_metrics)

        if (ep + 1) % log_interval == 0:
            avg_reward = sum(h["reward"] for h in history[-log_interval:]) / log_interval
            avg_disc = sum(h["discounted_return"] for h in history[-log_interval:]) / log_interval
            success_rate = sum(h["success"] for h in history[-log_interval:]) / log_interval
            print(
                f"Episode {ep + 1}/{num_episodes} | "
                f"avg_reward={avg_reward:.1f} | "
                f"avg_discounted_return={avg_disc:.1f} | "
                f"success_rate={success_rate:.1%}"
            )

    return history
