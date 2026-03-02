"""
Сохранение и загрузка параметров оценки (CSV).
"""


def save_params(path: str, params: dict) -> None:
    """
    Сохранить параметры оценки в файл (CSV).
    
    Input:
        path — путь к файлу (например "eval_params.json")
        params — словарь с параметрами оценки:
            model_path (str) — путь к файлу модели
            num_episodes (int) — количество эпизодов для оценки
            render (bool) — визуализировать ли ход
            num_disks (int) — количество дисков
            agent_method (str) — метод агента
            save_results (str | None) — путь для сохранения результатов
    Output: None
    Raises: IOError при ошибке записи
    """
    ...


def load_params(path: str) -> dict:
    """
    Загрузить параметры оценки из JSON-файла.
    
    Input:
        path — путь к файлу с сохранёнными параметрами
    Output: dict с ключами model_path, num_episodes, render, num_disks, agent_method, ...
    Raises: FileNotFoundError если файл не найден, json.JSONDecodeError при ошибке парсинга
    """
    ...
