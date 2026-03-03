# FILE: ./run/train.py
import sys
import argparse
from pathlib import Path

# Добавляем корень проекта в путь
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings
from env.environment import create_env
from env.rewards import Reward
from env.actions import get_action_space
from agent import create_agent
from training.trainer import train
from training.logger import setup_logger, log_message
from utils.params import save_history

def parse_args() -> object:
    parser = argparse.ArgumentParser(description="Train Tower of Hanoi agent")
    parser.add_argument("--num_episodes", type=int, default=settings.NUM_EPISODES,
                        help="Number of episodes to run")
    parser.add_argument("--max_steps", type=int, default=settings.MAX_STEPS_PER_EPISODE,
                        help="Step limit per episode")
    parser.add_argument("--random_init", action="store_true", default=settings.RANDOM_INIT,
                        help="Start each episode from a random valid state")
    parser.add_argument("--no-random_init", action="store_false", dest="random_init",
                        help="Disable random initialization")
    parser.add_argument("--checkpoint_interval", type=int, default=settings.CHECKPOINT_INTERVAL,
                        help="Episodes between checkpoints")
    parser.add_argument("--load_model", type=str, default=None,
                        help="Path to model to load for continued training (e.g. model.pth or model_checkpoint_5000.pth)")
    parser.add_argument("--save_model", type=str, default="model.pth",
                        help="Path to save the trained model (default: model.pth)")
    parser.add_argument("--entropy_adaptive", action="store_true",
                        help="Adaptive entropy: coef decreases as avg steps decrease")
    parser.add_argument("--no-entropy_adaptive", action="store_true",
                        help="Disable adaptive entropy (use fixed coef)")
    parser.add_argument("--reward_step", type=float, default=None,
                        help="Override REWARD_STEP from config")
    parser.add_argument("--reward_goal", type=float, default=None,
                        help="Override REWARD_GOAL from config")
    parser.add_argument("--reward_invalid_move", type=float, default=None,
                        help="Override REWARD_INVALID_MOVE from config")
    parser.add_argument("--reward_correct_placement", type=float, default=None,
                        help="Override REWARD_CORRECT_PLACEMENT from config")
    parser.add_argument("--entropy_coef", type=float, default=None,
                        help="Override REINFORCE_ENTROPY_COEF (fixed, when not adaptive)")
    parser.add_argument("--entropy_coef_min", type=float, default=None,
                        help="Override REINFORCE_ENTROPY_COEF_MIN")
    parser.add_argument("--entropy_coef_max", type=float, default=None,
                        help="Override REINFORCE_ENTROPY_COEF_MAX")
    parser.add_argument("--entropy_window", type=int, default=None,
                        help="Override REINFORCE_ENTROPY_WINDOW")
    parser.add_argument("--log_interval", type=int, default=None,
                        help="Override LOG_INTERVAL from config")
    parser.add_argument("--history_path", type=str, default=None,
                        help="Path to save training history (default: logs/training_history.json)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Переопределение параметров из CLI
    if args.reward_step is not None:
        settings.REWARD_STEP = args.reward_step
    if args.reward_goal is not None:
        settings.REWARD_GOAL = args.reward_goal
    if args.reward_invalid_move is not None:
        settings.REWARD_INVALID_MOVE = args.reward_invalid_move
    if args.reward_correct_placement is not None:
        settings.REWARD_CORRECT_PLACEMENT = args.reward_correct_placement
    if args.entropy_coef is not None:
        settings.REINFORCE_ENTROPY_COEF = args.entropy_coef
    if args.entropy_coef_min is not None:
        settings.REINFORCE_ENTROPY_COEF_MIN = args.entropy_coef_min
    if args.entropy_coef_max is not None:
        settings.REINFORCE_ENTROPY_COEF_MAX = args.entropy_coef_max
    if args.entropy_window is not None:
        settings.REINFORCE_ENTROPY_WINDOW = args.entropy_window
    if args.log_interval is not None:
        settings.LOG_INTERVAL = args.log_interval
    if args.no_entropy_adaptive:
        settings.REINFORCE_ENTROPY_ADAPTIVE = False

    # 1. Настройка логгера (из settings.py)
    setup_logger(
        log_file=settings.LOG_FILE,
        level=settings.LOG_LEVEL
    )
    
    entropy_adaptive = (args.entropy_adaptive or getattr(settings, "REINFORCE_ENTROPY_ADAPTIVE", False)) and not args.no_entropy_adaptive
    if args.no_entropy_adaptive:
        settings.REINFORCE_ENTROPY_ADAPTIVE = False
    elif args.entropy_adaptive:
        settings.REINFORCE_ENTROPY_ADAPTIVE = True

    log_message(f"=== Starting Training Session ===")
    log_message(f"Method: {settings.AGENT_METHOD} | Disks: {settings.NUM_DISKS}")
    if entropy_adaptive:
        log_message(f"Entropy adaptive: min={getattr(settings, 'REINFORCE_ENTROPY_COEF_MIN', 0.01)}, max={getattr(settings, 'REINFORCE_ENTROPY_COEF_MAX', 0.2)}")

    # 2. Инициализация среды
    reward_scheme = Reward.from_config(settings)
    env = create_env(
        num_disks=settings.NUM_DISKS,
        num_sticks=settings.NUM_STICKS,
        max_steps=args.max_steps,
        reward=reward_scheme
    )
    
    # 3. Настройка агента
    obs_dim = settings.NUM_DISKS * 2
    action_space = get_action_space(settings.NUM_STICKS)
    
    # Собираем конфиг агента из всех доступных параметров settings.py
    entropy_adaptive = args.entropy_adaptive or getattr(settings, "REINFORCE_ENTROPY_ADAPTIVE", False)
    agent_config = {
        "learning_rate": settings.REINFORCE_LR,
        "discount_factor": settings.DISCOUNT_FACTOR,
        "gamma": settings.GAMMA,
        "hidden_dims": settings.REINFORCE_HIDDEN_DIMS,
        "entropy_coef": getattr(settings, "REINFORCE_ENTROPY_COEF_MAX", 0.2) if entropy_adaptive else getattr(settings, "REINFORCE_ENTROPY_COEF", 0.1),
        "value_lr": getattr(settings, "REINFORCE_BASELINE_VALUE_LR", 1e-2),
        "max_kl": getattr(settings, "TRPO_MAX_KL", 0.01),
    }

    agent = create_agent(settings.AGENT_METHOD, obs_dim, action_space, agent_config)

    # Загрузка весов для дообучения
    if args.load_model:
        agent.load(args.load_model)
        log_message(f"Loaded model from {args.load_model} for continued training")
    
    # 4. Запуск обучения
    try:
        history = train(
            env=env,
            agent=agent,
            num_episodes=args.num_episodes,
            max_steps_per_episode=args.max_steps,
            log_interval=args.log_interval or settings.LOG_INTERVAL,
            random_init=args.random_init,
            checkpoint_interval=args.checkpoint_interval,
            entropy_adaptive=entropy_adaptive,
            entropy_coef_min=args.entropy_coef_min if args.entropy_coef_min is not None else getattr(settings, "REINFORCE_ENTROPY_COEF_MIN", 0.01),
            entropy_coef_max=args.entropy_coef_max if args.entropy_coef_max is not None else getattr(settings, "REINFORCE_ENTROPY_COEF_MAX", 0.2),
            entropy_window=args.entropy_window if args.entropy_window is not None else getattr(settings, "REINFORCE_ENTROPY_WINDOW", 100),
        )
        
        # 5. Сохранение результатов
        # Сохраняем веса нейросети
        agent.save(args.save_model)
        log_message(f"Model weights saved to {args.save_model}")
        
        # Сохраняем историю для графиков
        history_path = args.history_path or "logs/training_history.json"
        save_history(history_path, history)
        log_message(f"Training history saved to {history_path}")
        
        log_message("=== Training Successfully Finished ===")
        
    except KeyboardInterrupt:
        log_message("Training interrupted by user. Metrics and model were NOT saved.", level="WARNING")
    except Exception as e:
        log_message(f"Critical error during training: {e}", level="ERROR")
        raise e

if __name__ == "__main__":
    main()