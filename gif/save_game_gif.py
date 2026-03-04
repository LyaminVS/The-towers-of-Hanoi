"""
Скрипт сохранения одной игры в GIF.
Запуск: python gif/save_game_gif.py --output game.gif [--load_model model.pth]
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if "--headless" in sys.argv:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import argparse
import pygame
from PIL import Image

from config import settings
from env.environment import create_env
from env.rewards import Reward
from env.actions import get_action_space, get_valid_actions
from env.render import PygameRenderer
from agent import create_agent, create_baseline


def parse_args():
    parser = argparse.ArgumentParser(description="Save one game as GIF")
    parser.add_argument("--output", "-o", type=str, default="game.gif")
    parser.add_argument("--load_model", type=str, default=settings.EVAL_MODEL_PATH)
    parser.add_argument("--num_disks", type=int, default=settings.NUM_DISKS)
    parser.add_argument("--agent_method", type=str, default=settings.AGENT_METHOD,
                        choices=["reinforce", "trpo"])
    parser.add_argument("--value_estimator", type=str, default=settings.VALUE_ESTIMATOR,
                        choices=["zero", "tabular"])
    parser.add_argument("--duration", type=int, default=400)
    parser.add_argument("--loop", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def run_one_episode_and_capture(env, agent, renderer) -> list:
    frames = []
    observation, info = env.reset()
    total_reward = 0.0
    step_count = 0

    while True:
        state_tuple = tuple(tuple(d) for d in info["state"])
        valid_actions = get_valid_actions(state_tuple, env.num_sticks)

        renderer.render(state_tuple, step_count, total_reward, message=f"Step {step_count}")
        img = Image.frombytes("RGB", (renderer.width, renderer.height), renderer.get_surface_bytes())
        frames.append(img)

        action, _ = agent.select_action(observation, valid_actions, training=False)
        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        step_count = info["step_count"]

        if terminated or truncated:
            break

    success = info.get("is_correct_placement", 0) == env.num_disks
    final_state = tuple(tuple(d) for d in info["state"])
    msg = "Perfect!" if success else "Done"
    renderer.render(final_state, step_count, total_reward, message=msg)
    img = Image.frombytes("RGB", (renderer.width, renderer.height), renderer.get_surface_bytes())
    frames.append(img)

    return frames


def main():
    args = parse_args()

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
        "entropy_coef": getattr(settings, "REINFORCE_ENTROPY_COEF", 0.01),
        "max_kl": getattr(settings, "TRPO_MAX_KL", 0.01),
        "num_disks": args.num_disks,
        "num_sticks": settings.NUM_STICKS,
        "history_len": getattr(settings, "HISTORY_LEN", 20),
    }

    baseline = create_baseline(args.value_estimator, agent_config)
    agent = create_agent(args.agent_method, obs_dim, action_space, agent_config, baseline)
    agent.load(args.load_model)

    pygame.init()
    renderer = PygameRenderer(args.num_disks, settings.NUM_STICKS)

    try:
        frames = run_one_episode_and_capture(env, agent, renderer)
    finally:
        renderer.close()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=args.duration,
        loop=args.loop,
    )

    print(f"Saved {len(frames)} frames to {out_path}")


if __name__ == "__main__":
    main()
