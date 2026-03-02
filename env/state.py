"""
State representation for Tower of Hanoi.

State:
    tuple of pairs (stick, height)
    ordered from largest disk to smallest disk.

Rules:
    - height = 0 → disk is at the bottom of its stick
    - heights on each stick must be consecutive starting from 0
"""

from typing import Tuple, List
import random
import torch

State = Tuple[Tuple[int, int], ...]


def get_random_valid_state(num_disks: int, num_sticks: int = 3, exclude_terminal: bool = True) -> State:
    """
    Случайное валидное состояние: диски случайно распределены по палкам,
    на каждой палке — правильный порядок (больший снизу).
    State[i] = (stick, height) для диска i.
    exclude_terminal — не возвращать целевое состояние (все на последней палке)
    """
    if num_disks <= 0:
        raise ValueError("num_disks must be positive")
    for _ in range(100):
        sticks: List[List[int]] = [[] for _ in range(num_sticks)]
        for disk_id in range(num_disks):
            stick = random.randint(0, num_sticks - 1)
            sticks[stick].append(disk_id)
        state: List[Tuple[int, int]] = [None] * num_disks
        for stick_idx, disk_ids in enumerate(sticks):
            disk_ids.sort()
            for height, disk_id in enumerate(disk_ids):
                state[disk_id] = (stick_idx, height)
        s = tuple(state)
        if exclude_terminal and is_terminal_state(s, num_disks):
            continue
        return s
    return get_initial_state(num_disks)


def get_initial_state(num_disks: int) -> State:
    """
    All disks start on stick 0.
    Largest disk at bottom (height 0).
    Smallest disk at top.
    """
    if num_disks <= 0:
        raise ValueError("num_disks must be positive")

    state: List[Tuple[int, int]] = []

    for i in range(num_disks):
        # disks ordered from largest to smallest
        height = i  # largest gets 0, next 1, ..., smallest gets num_disks-1
        state.append((0, height))

    return tuple(state)


def state_to_observation(state: State) -> torch.Tensor:
    """
    Convert state into flat torch vector:
    ((stick_0, height_0), (stick_1, height_1), ...)  → 
    [stick_0, height_0, stick_1, height_1, ...]
    No normalization.
    """

    obs = []

    for stick, height in state:
        obs.append(stick)
        obs.append(height)

    return torch.tensor(obs, dtype=torch.float32)


def observation_to_state(observation) -> State:
    """
    Convert flat observation tensor back to state:
    [stick_0, height_0, stick_1, height_1, ...]
    → ((stick_0, height_0), (stick_1, height_1), ...)
    """

    # если пришёл torch.Tensor
    if isinstance(observation, torch.Tensor):
        observation = observation.tolist()

    if len(observation) % 2 != 0:
        raise ValueError("Observation length must be even.")

    state = []

    for i in range(0, len(observation), 2):
        stick = int(observation[i])
        height = int(observation[i + 1])
        state.append((stick, height))

    return tuple(state)


def is_terminal_state(state: State, num_disks: int) -> bool:
    """
    Terminal iff all disks are on stick 2 AND stacked in the correct order:

    State is ordered from largest to smallest disk.
    height=0 is the bottom of the stick.

    So terminal state must be:
        state[i] == (2, i) for all i in 0..num_disks-1
    """
    if len(state) != num_disks:
        raise ValueError("State length mismatch")

    for i, (stick, height) in enumerate(state):
        if stick != 2:
            return False
        if height != i:
            return False

    return True





