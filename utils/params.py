# FILE: ./utils/params.py
import json
import os
import random
import numpy as np
import torch

def save_history(path: str, history: list[dict]) -> None:
    """Сохранить историю обучения (список метрик) в JSON файл."""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=4)
        print(f"Successfully saved training history to {path}")
    except Exception as e:
        print(f"Error saving history: {e}")


def save_eval_params(path: str, params: dict) -> None:
    """Сохранить параметры оценки в JSON файл."""
    try:
        # Убеждаемся, что директория существует
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(params, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving params to {path}: {e}")

def load_eval_params(path: str) -> dict:
    """Загрузить параметры оценки из JSON файла."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Params file not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
def set_seed(seed: int):
    """Фиксация всех генераторов случайных чисел для воспроизводимости."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Для полной детерминированности (может немного замедлить обучение)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")