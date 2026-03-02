"""
Логирование в файл и консоль.
"""

import logging
_LOGGER: logging.Logger | None = None

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _get_logger() -> logging.Logger:
    """Возвращает настроенный логгер или создаёт дефолтный по умолчанию."""
    global _LOGGER
    if _LOGGER is None:
        _LOGGER = logging.getLogger("tower_hanoi")
        if not _LOGGER.handlers:
            _LOGGER.setLevel(logging.INFO)
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
            _LOGGER.addHandler(h)
    return _LOGGER


def setup_logger(
    log_file: str | None = None,
    console: bool = True,
    level: str = "INFO",
) -> logging.Logger:
    """
    Настроить логгер с выводом в консоль и/или в файл.

    Input:
        log_file — путь к файлу (None — не писать в файл)
        console — выводить ли в stdout
        level — уровень логирования: "DEBUG", "INFO", "WARNING", "ERROR"
    Output: настроенный logger (logging.Logger)
    """
    global _LOGGER
    _LOGGER = logging.getLogger("tower_hanoi")
    _LOGGER.setLevel(getattr(logging, level.upper(), logging.INFO))
    _LOGGER.handlers.clear()

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    if console:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        _LOGGER.addHandler(ch)

    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(formatter)
        _LOGGER.addHandler(fh)

    return _LOGGER


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
    logger = _get_logger()
    parts = [f"episode={episode}", f"reward={reward:.2f}", f"steps={steps}", f"success={success}"]
    for k, v in kwargs.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.4f}")
        else:
            parts.append(f"{k}={v}")
    logger.info(" | ".join(parts))


def log_metrics(metrics: dict, prefix: str = "") -> None:
    """
    Записать словарь метрик одной строкой.

    Input:
        metrics — dict {name: value}, значения должны быть сериализуемы
        prefix — префикс для ключей (например "episode_" -> "episode_reward")
    Output: None
    """
    logger = _get_logger()
    pairs = []
    for k, v in metrics.items():
        key = f"{prefix}{k}" if prefix else k
        if isinstance(v, float):
            pairs.append(f"{key}={v:.4f}")
        else:
            pairs.append(f"{key}={v}")
    logger.info(" | ".join(pairs))


def log_message(message: str, level: str = "INFO") -> None:
    """
    Записать произвольное сообщение.

    Input:
        message — текст сообщения
        level — "DEBUG" | "INFO" | "WARNING" | "ERROR"
    Output: None
    """
    logger = _get_logger()
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.log(log_level, "%s", message)
