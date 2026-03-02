"""
Логирование в файл и консоль.
"""


def setup_logger(
    log_file: str | None = None,
    console: bool = True,
    level: str = "INFO",
) -> object:
    """
    Настроить логгер с выводом в консоль и/или в файл.
    
    Input:
        log_file — путь к файлу (None — не писать в файл)
        console — выводить ли в stdout
        level — уровень логирования: "DEBUG", "INFO", "WARNING", "ERROR"
    Output: настроенный logger (logging.Logger)
    """
    ...


def log_episode(episode: int, reward: float, steps: int, success: bool, **kwargs) -> None:
    """
    Записать метрики эпизода.
    
    Input:
        episode — номер эпизода
        reward — суммарная награда
        steps — количество шагов
        success — достигнута ли цель
        **kwargs — дополнительные метрики (policy_loss, value_loss, kl, ...)
    Output: None
    """
    ...


def log_metrics(metrics: dict, prefix: str = "") -> None:
    """
    Записать словарь метрик одной строкой.
    
    Input:
        metrics — dict {name: value}, значения должны быть сериализуемы
        prefix — префикс для ключей (например "episode_" → "episode_reward")
    Output: None
    """
    ...


def log_message(message: str, level: str = "INFO") -> None:
    """
    Записать произвольное сообщение.
    
    Input:
        message — текст сообщения
        level — "DEBUG" | "INFO" | "WARNING" | "ERROR"
    Output: None
    """
    ...
