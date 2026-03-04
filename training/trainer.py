# FILE: ./training/trainer.py
import torch
from env.actions import get_valid_actions
from training.logger import log_episode, log_message

def run_episode(env, agent, max_steps: int, random_init: bool = False, update_agent: bool = True):
    """Один эпизод. Если update_agent=False, update() не вызывается (для батча)."""
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

    update_metrics = agent.update() if update_agent else None
    success = info.get("is_correct_placement", 0) == env.num_disks
    return total_reward, info["step_count"], success, rewards, update_metrics

def _compute_adaptive_entropy_coef(
    avg_steps: float,
    steps_max: float,
    coef_min: float,
    coef_max: float,
) -> float:
    """Линейная интерполяция: меньше шагов -> min coef, больше -> max coef."""
    if steps_max <= 0:
        return coef_min
    t = avg_steps / steps_max
    t = max(0.0, min(1.0, t))
    return coef_min + (coef_max - coef_min) * t


def train(
    env,
    agent,
    num_episodes,
    max_steps_per_episode,
    log_interval=100,
    random_init=False,
    checkpoint_interval=1000,
    entropy_adaptive=False,
    entropy_coef_min=0.01,
    entropy_coef_max=0.2,
    entropy_window=100,
    episodes_per_update=1,
):
    gamma = agent.config.get("discount_factor", 0.99)
    history = []
    steps_buffer = []
    batch_mode = episodes_per_update > 1 and hasattr(agent, "update_batch")

    if hasattr(agent, "env") and agent.env is None:
        agent.env = env

    for ep in range(num_episodes):
        if batch_mode:
            total_reward, num_steps, success, rewards, _ = run_episode(
                env, agent, max_steps_per_episode, random_init=random_init, update_agent=False
            )
            traj = agent.get_trajectory()
            agent.reset_trajectory()
            if not hasattr(agent, "_batch_trajectories"):
                agent._batch_trajectories = []
                agent._batch_episode_metrics = []
            agent._batch_trajectories.append(traj)
            agent._batch_episode_metrics.append({
                "reward": total_reward, "steps": num_steps, "success": success, "rewards": rewards,
            })
            if len(agent._batch_trajectories) < episodes_per_update:
                continue
            update_metrics = agent.update_batch(agent._batch_trajectories)
            agent._batch_trajectories = []
            batch_metrics = getattr(agent, "_batch_episode_metrics", [])
            agent._batch_episode_metrics = []
        else:
            total_reward, num_steps, success, rewards, update_metrics = run_episode(
                env, agent, max_steps_per_episode, random_init=random_init, update_agent=True
            )
            batch_metrics = [{"reward": total_reward, "steps": num_steps, "success": success, "rewards": rewards}]

        # Адаптивный коэффициент энтропии
        if entropy_adaptive and hasattr(agent, "entropy_coef"):
            for m in batch_metrics:
                steps_buffer.append(m["steps"])
            if len(steps_buffer) > entropy_window:
                steps_buffer = steps_buffer[-entropy_window:]
            avg_steps = sum(steps_buffer) / len(steps_buffer)
            agent.entropy_coef = _compute_adaptive_entropy_coef(
                avg_steps, steps_max=float(max_steps_per_episode),
                coef_min=entropy_coef_min, coef_max=entropy_coef_max,
            )

        for i, m in enumerate(batch_metrics):
            discounted_return = sum(gamma**k * r for k, r in enumerate(m["rewards"]))
            episode_metrics = {
                "episode": (ep - len(batch_metrics) + i + 2) if batch_mode else (ep + 1),
                "reward": m["reward"],
                "steps": m["steps"],
                "success": int(m["success"]),
                "discounted_return": discounted_return,
            }
            if update_metrics:
                episode_metrics.update(update_metrics)
            if hasattr(agent, "entropy_coef"):
                episode_metrics["entropy_coef"] = agent.entropy_coef
            history.append(episode_metrics)

        ep_done = ep + 1
        if (ep_done) % log_interval == 0:
            recent = history[-log_interval:]
            n_recent = len(recent) or 1
            avg_reward = sum(h["reward"] for h in recent) / n_recent
            success_rate = sum(h["success"] for h in recent) / n_recent
            log_kw = {
                "success_rate": f"{success_rate:.1%}",
                "loss": update_metrics.get("policy_loss", 0) if update_metrics else 0,
                "entropy": update_metrics.get("entropy", 0) if update_metrics else 0,
            }
            if "entropy_coef" in episode_metrics:
                log_kw["entropy_coef"] = episode_metrics["entropy_coef"]
            if update_metrics and "policy_update_accepted" in update_metrics:
                log_kw["accepted"] = update_metrics["policy_update_accepted"]
            log_episode(
                episode=ep + 1,
                reward=avg_reward,
                steps=int(sum(h["steps"] for h in recent) / n_recent),
                success=success_rate > 0.5,
                **log_kw,
            )
        
        # Сохранять модель каждые checkpoint_interval эпизодов
        if (ep + 1) % checkpoint_interval == 0:
            checkpoint_path = f"model_checkpoint_{ep + 1}.pth"
            agent.save(checkpoint_path)
            log_message(f"Checkpoint saved: {checkpoint_path}")

    return history # Возвращаем накопленные данные