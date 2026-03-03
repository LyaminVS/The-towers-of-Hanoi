"""
Выбор device для PyTorch (CPU/GPU).
"""

import torch


def get_device(device=None):
    """
    Возвращает torch.device для вычислений.

    device: str | None
        - None или "auto" — cuda, если доступен, иначе cpu
        - "cpu" — CPU
        - "cuda" или "cuda:0" — GPU (первая по умолчанию)
    """
    if device is None or (isinstance(device, str) and device.lower() == "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
