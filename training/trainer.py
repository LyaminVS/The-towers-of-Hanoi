# FILE: ./training/trainer.py
import torch
from env.actions import get_valid_actions
from training.logger import log_episode, log_message

def run_episode(env, agent, max_steps: int, use_death_penalty: bool = False, random_init: bool = False):
    env.reward.use_death_penalty = use_death_penalty
    env.max_steps = max_steps
    agent.reset_trajectory()

    observation, info = env.reset(random_init=random_init)
    total_reward = 0.0
    rewards = []

    while True:
        current_state_tuple = tuple(tuple(d) for d in info["state"])
        valid_actions = get_valid_actions(current_state_tuple, env.num_sticks)
        
        state_before = observation
        action, log_prob = agent.select_action(observation, valid_actions, training=True)

        observation, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        agent.store_transition(state_before, action, reward, log_prob)

        total_reward += reward
        if terminated or truncated:
            break

    update_metrics = agent.update()
    success = info.get("is_correct_placement", 0) == env.num_disks
    return total_reward, info["step_count"], success, rewards, update_metrics

def _compute_adaptive_entropy_coef(
    avg_steps: float,
    steps_min: float,
    steps_max: float,
    coef_min: float,
    coef_max: float,
) -> float:
    """Линейная интерполяция: много шагов -> max coef, мало шагов -> min coef."""
    if steps_max <= steps_min:
        return coef_max
    t = (avg_steps - steps_min) / (steps_max - steps_min)
    t = max(0.0, min(1.0, t))
    return coef_min + (coef_max - coef_min) * t


def train(
    env,
    agent,
    num_episodes,
    max_steps_per_episode,
    use_death_penalty=False,
    log_interval=100,
    random_init=False,
    checkpoint_interval=1000,
    entropy_adaptive=False,
    entropy_coef_min=0.01,
    entropy_coef_max=0.2,
    entropy_window=100,
):
    gamma = agent.config.get("discount_factor", 0.99)
    history = []
    steps_buffer = []  # для скользящего среднего
    steps_optimal = 2 ** env.num_disks - 1  # оптимальное число шагов

    for ep in range(num_episodes):
        total_reward, num_steps, success, rewards, update_metrics = run_episode(
            env, agent, max_steps_per_episode, use_death_penalty, random_init=random_init
        )

        # Адаптивный коэффициент энтропии: уменьшается при уменьшении ср. шагов
        if entropy_adaptive and hasattr(agent, "entropy_coef"):
            steps_buffer.append(num_steps)
            if len(steps_buffer) > entropy_window:
                steps_buffer.pop(0)
            avg_steps = sum(steps_buffer) / len(steps_buffer)
            agent.entropy_coef = _compute_adaptive_entropy_coef(
                avg_steps,
                steps_min=steps_optimal,
                steps_max=float(max_steps_per_episode),
                coef_min=entropy_coef_min,
                coef_max=entropy_coef_max,
            )

        # Считаем дисконтированную награду для графиков (G_0)
        discounted_return = sum(gamma**k * r for k, r in enumerate(rewards))

        # Собираем полный словарь метрик для этого эпизода
        episode_metrics = {
            "episode": ep + 1,
            "reward": total_reward,
            "steps": num_steps,
            "success": int(success),
            "discounted_return": discounted_return,
        }
        if update_metrics:
            episode_metrics.update(update_metrics)
        if hasattr(agent, "entropy_coef"):
            episode_metrics["entropy_coef"] = agent.entropy_coef
        history.append(episode_metrics)

        if (ep + 1) % log_interval == 0:
            recent = history[-log_interval:]
            avg_reward = sum(h["reward"] for h in recent) / log_interval
            success_rate = sum(h["success"] for h in recent) / log_interval
            log_kw = {
                "success_rate": f"{success_rate:.1%}",
                "loss": update_metrics.get("policy_loss", 0) if update_metrics else 0,
                "entropy": update_metrics.get("entropy", 0) if update_metrics else 0,
            }
            if "entropy_coef" in episode_metrics:
                log_kw["entropy_coef"] = episode_metrics["entropy_coef"]
            log_episode(
                episode=ep + 1,
                reward=avg_reward,
                steps=int(sum(h["steps"] for h in recent) / log_interval),
                success=success_rate > 0.5,
                **log_kw,
            )
        
        # Сохранять модель каждые checkpoint_interval эпизодов
        if (ep + 1) % checkpoint_interval == 0:
            checkpoint_path = f"model_checkpoint_{ep + 1}.pth"
            agent.save(checkpoint_path)
            log_message(f"Checkpoint saved: {checkpoint_path}")

    return history # Возвращаем накопленные данные