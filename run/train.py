"""
Скрипт обучения агента. Запуск: python run/train.py [--args]
"""


def parse_args() -> object:
    """
    Парсинг аргументов командной строки (argparse).
    
    Input: —
    Output: объект (Namespace) с полями:
        num_disks — количество дисков
        num_episodes — количество эпизодов
        agent_method — метод агента (reinforce, reinforce_baseline, trpo)
        save_model — путь для сохранения модели (None — не сохранять)
        use_death_penalty — использовать ли штраф смерти
        log_interval — интервал логирования
        log_file — путь к файлу логов (None — только консоль)
        и др.
    """
    ...


def main() -> None:
    """
    Запуск обучения: создать среду, агента, вызвать training.train().
    
    Порядок: parse_args → create_env → create_agent → train → agent.save (если save_model)
    
    Input: —
    Output: None (сохраняет модель при указании --save_model)
    """
    ...


if __name__ == "__main__":
    main()
