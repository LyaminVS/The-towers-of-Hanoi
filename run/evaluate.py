"""
Скрипт оценки обученного агента. Запуск: python run/evaluate.py [--args]
С визуализацией: python run/evaluate.py --render --load_model model.pth
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import pygame
from config import settings
from env.environment import create_env
from env.rewards import Reward
from env.actions import get_action_space, get_valid_actions
from env.render import PygameRenderer
from agent import create_agent


def parse_args() -> object:
    """
    Парсинг аргументов командной строки.
    """
    parser = argparse.ArgumentParser(description="Evaluate trained Tower of Hanoi agent")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--load_model", type=str, default=settings.EVAL_MODEL_PATH)
    parser.add_argument("--num_disks", type=int, default=settings.NUM_DISKS)
    parser.add_argument("--agent_method", type=str, default=settings.AGENT_METHOD,
                        choices=["reinforce", "reinforce_baseline", "trpo"])
    parser.add_argument("--num_episodes", type=int, default=settings.EVAL_NUM_EPISODES)
    parser.add_argument("--render", action="store_true", default=settings.EVAL_RENDER,
                        help="Визуализировать ход агента (Pygame)")
    parser.add_argument("--delay_ms", type=int, default=400,
                        help="Задержка между шагами при --render (мс)")
    parser.add_argument("--sample", action="store_true", help="Использовать сэмплирование при оценке вместо argmax")
    parser.add_argument("--save_results", type=str, default=settings.EVAL_SAVE_RESULTS)
    return parser.parse_args()


def evaluate(env, agent, num_episodes: int = 10, render: bool = False, delay_ms: int = 400, sample: bool = False) -> dict:
    """
    Оценка обученного агента без exploration (greedy действия).
    
    Input:
        env — среда TowerOfHanoiEnv
        agent — агент с загруженной моделью (training=False для select_action)
        num_episodes — количество эпизодов для оценки
        render — если True, показывать Pygame-визуализацию
        delay_ms — задержка между шагами при render (мс)
    Output: dict со средними метриками
    """
    rewards_list = []
    steps_list = []
    successes = []

    renderer = PygameRenderer(env.num_disks, env.num_sticks) if render else None

    for ep in range(num_episodes):
        observation, info = env.reset()
        total_reward = 0.0
        step_count = 0

        while True:
            state_tuple = tuple(tuple(d) for d in info["state"])
            valid_actions = get_valid_actions(state_tuple, env.num_sticks)

            if render and renderer:
                renderer.render(
                    state_tuple,
                    step_count,
                    total_reward,
                    selected_stick=None,
                    message=f"Episode {ep+1}/{num_episodes}",
                )
                for _ in range(max(1, delay_ms // 16)):
                    for e in pygame.event.get():
                        if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                            if renderer:
                                renderer.close()
                            return {
                                "mean_reward": sum(rewards_list) / len(rewards_list) if rewards_list else 0,
                                "mean_steps": sum(steps_list) / len(steps_list) if steps_list else 0,
                                "success_rate": sum(successes) / len(successes) if successes else 0,
                                "episodes_completed": len(rewards_list),
                            }
                    renderer.clock.tick(60)

            # training flag controls sampling vs greedy inside select_action
            use_sampling = sample or settings.EVAL_SAMPLE
            action, _ = agent.select_action(observation, valid_actions, training=use_sampling)
            observation, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            step_count = info["step_count"]

            if terminated or truncated:
                break

        success = info.get("is_correct_placement", 0) == env.num_disks
        rewards_list.append(total_reward)
        steps_list.append(step_count)
        successes.append(success)

        if render and renderer:
            final_state = tuple(tuple(d) for d in info["state"])
            msg = "Perfect!" if success else "Truncated"
            renderer.render(final_state, step_count, total_reward, message=msg)
            pygame.time.wait(delay_ms * 2)

    if renderer:
        renderer.close()

    return {
        "mean_reward": sum(rewards_list) / num_episodes,
        "mean_steps": sum(steps_list) / num_episodes,
        "success_rate": sum(successes) / num_episodes,
        "episodes_completed": num_episodes,
    }


def main() -> None:
    args = parse_args()

    # Установка seed для воспроизводимости
    import numpy as np
    import torch
    import random
    
    if args.seed is not None:
        settings.SEED = args.seed
    
    seed = settings.SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    reward_scheme = Reward.from_config(settings)

    env = create_env(
        num_disks=args.num_disks,
        num_sticks=settings.NUM_STICKS,
        max_steps=settings.MAX_STEPS_PER_EPISODE,
        reward=reward_scheme,
    )

    obs_dim = args.num_disks * 2
    action_space = get_action_space(settings.NUM_STICKS)
    agent_config = {
        "learning_rate": settings.REINFORCE_LR,
        "discount_factor": settings.DISCOUNT_FACTOR,
        "gamma": settings.GAMMA,
        "hidden_dims": settings.REINFORCE_HIDDEN_DIMS,
        "value_lr": getattr(settings, "REINFORCE_BASELINE_VALUE_LR", 1e-2),
        "max_kl": getattr(settings, "TRPO_MAX_KL", 0.01),
    }

    agent = create_agent(args.agent_method, obs_dim, action_space, agent_config)
    agent.load(args.load_model)

    print(f"Evaluating: {args.load_model} | {args.num_episodes} episodes | render={args.render}")

    metrics = evaluate(
        env, agent,
        num_episodes=args.num_episodes,
        render=args.render,
        delay_ms=args.delay_ms,
        sample=args.sample,
    )

    print(f"\n=== Results ===")
    print(f"Mean reward:    {metrics['mean_reward']:.2f}")
    print(f"Mean steps:     {metrics['mean_steps']:.1f}")
    print(f"Success rate:  {metrics['success_rate']:.1%}")

    if args.save_results:
        import json
        with open(args.save_results, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved to {args.save_results}")


if __name__ == "__main__":
    main()
