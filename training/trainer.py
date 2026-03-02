# FILE: ./training/trainer.py
import torch
from env.actions import get_valid_actions
from training.logger import log_episode, log_message

def run_episode(
    env,
    agent,
    max_steps: int,
    use_death_penalty: bool = False,
) -> tuple[float, int, bool, list[float], dict]:
    # Устанавливаем параметры в среду
    env.reward.use_death_penalty = use_death_penalty
    env.max_steps = max_steps
    agent.reset_trajectory()

    observation, info = env.reset()
    total_reward = 0.0
    rewards: list[float] = []

    while True:
        # Важно: передаем валидные действия для маскировки
        current_state_tuple = tuple(tuple(d) for d in info["state"])
        valid_actions = get_valid_actions(current_state_tuple, env.num_sticks)
        
        state_before = observation
        action, log_prob = agent.select_action(observation, valid_actions, training=True)

        observation, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        agent.store_transition(state_before, action, reward, log_prob)

        total_reward += reward
        num_steps = info["step_count"]
        
        if terminated or truncated:
            break

    update_metrics = agent.update()
    success = info.get("is_correct_placement", 0) == env.num_disks
    return total_reward, num_steps, success, rewards, update_metrics

def _compute_discounted_return(rewards: list[float], gamma: float) -> float:
    return sum(gamma**k * r for k, r in enumerate(rewards))

def train(
    env,
    agent,
    num_episodes: int,
    max_steps_per_episode: int,
    use_death_penalty: bool = False, # ДОБАВЛЕНО
    log_interval: int = 100,
) -> list[dict]:
    gamma = agent.config.get("discount_factor", 0.99)
    history: list[dict] = []

    for ep in range(num_episodes):
        # Передаем use_death_penalty в run_episode
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
            recent = history[-log_interval:]
            avg_reward = sum(h["reward"] for h in recent) / log_interval
            avg_steps = sum(h["steps"] for h in recent) / log_interval
            success_rate = sum(h["success"] for h in recent) / log_interval
            
            # Используем ваш красивый логгер
            log_episode(
                episode=ep + 1,
                reward=avg_reward,
                steps=int(avg_steps),
                success=success_rate > 0.5,
                success_rate=f"{success_rate:.1%}",
                **update_metrics
            )

    return history