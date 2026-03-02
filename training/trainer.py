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

def train(env, agent, num_episodes, max_steps_per_episode, use_death_penalty=False, log_interval=100, random_init=False, checkpoint_interval=1000):
    gamma = agent.config.get("discount_factor", 0.99)
    history = [] # Сюда собираем всё

    for ep in range(num_episodes):
        total_reward, num_steps, success, rewards, update_metrics = run_episode(
            env, agent, max_steps_per_episode, use_death_penalty, random_init=random_init
        )
        
        # Считаем дисконтированную награду для графиков (G_0)
        discounted_return = sum(gamma**k * r for k, r in enumerate(rewards))

        # Собираем полный словарь метрик для этого эпизода
        episode_metrics = {
            "episode": ep + 1,
            "reward": total_reward,
            "steps": num_steps,
            "success": int(success), # 1 или 0 для графиков
            "discounted_return": discounted_return,
        }
        # Добавляем лоссы и прочее, что вернул агент (policy_loss, value_loss и т.д.)
        if update_metrics:
            episode_metrics.update(update_metrics)
            
        history.append(episode_metrics)

        if (ep + 1) % log_interval == 0:
            recent = history[-log_interval:]
            avg_reward = sum(h["reward"] for h in recent) / log_interval
            success_rate = sum(h["success"] for h in recent) / log_interval
            
            log_episode(
                episode=ep + 1,
                reward=avg_reward,
                steps=int(sum(h["steps"] for h in recent) / log_interval),
                success=success_rate > 0.5,
                success_rate=f"{success_rate:.1%}",
                loss=update_metrics.get("policy_loss", 0) if update_metrics else 0,
                entropy=update_metrics.get("entropy", 0) if update_metrics else 0,
            )
        
        # Сохранять модель каждые checkpoint_interval эпизодов
        if (ep + 1) % checkpoint_interval == 0:
            checkpoint_path = f"model_checkpoint_{ep + 1}.pth"
            agent.save(checkpoint_path)
            log_message(f"Checkpoint saved: {checkpoint_path}")

    return history # Возвращаем накопленные данные