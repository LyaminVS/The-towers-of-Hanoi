import pytest

from env.environment import create_env
from env.rewards import Reward
from env.state import get_initial_state
from config import settings


def test_invalid_move_penalty_and_state_unchanged():
    reward_scheme = Reward.from_config(settings)
    env = create_env(num_disks=3, num_sticks=3, max_steps=50, reward=reward_scheme)
    obs, info = env.reset()
    original_state = info["state"].copy()

    # Попытка хода с пустой палки -> считается invalid
    obs2, rew, terminated, truncated, info2 = env.step((1, 2))
    assert info2["is_invalid"] is True
    # оштрафованы шаг + invalid
    assert rew == settings.REWARD_STEP + settings.REWARD_INVALID_MOVE
    assert env._state == original_state
    assert not terminated

    # when no disk on last stick, bonus should be zero
    assert rew == settings.REWARD_STEP + settings.REWARD_INVALID_MOVE

    # Снова попробуем "большой на меньший" - сперва разместим диск на 0
    # move top disk (disk 2) с 0 на 2, потом попытаемся поставить диск 1 на 2
    env.reset()
    # сделаем допустимый ход 0->2
    _, rew1, _, _, _ = env.step((0, 2))
    assert rew1 == settings.REWARD_STEP
    # теперь возьмём ещё один диск сверху палки 0 и переместим на палку 2
    _, rew2, _, _, info3 = env.step((0, 2))
    # второй ход уже будет invalid, поскольку диск 1 больше диска 2
    assert info3["is_invalid"] is True
    assert rew2 == settings.REWARD_STEP + settings.REWARD_INVALID_MOVE
    # состояние не изменилось после invalid
    assert env._state[1][0] != 2


def test_goal_penalty():
    reward_scheme = Reward.from_config(settings)
    env = create_env(num_disks=2, num_sticks=3, max_steps=10, reward=reward_scheme)
    # ставим ручками конечное состояние
    env._state = list(get_initial_state(2))
    # поместим все диски на последнюю палку
    env._state[0] = (2, 0)
    env._state[1] = (2, 1)
    # проверим непосредственно схему наград для случая, когда цель достигнута
    rew, done = reward_scheme.compute(is_invalid=False, all_correct=True)
    assert rew == settings.REWARD_STEP + settings.REWARD_GOAL
    assert done is False


def test_correct_placement_bonus():
    # проверка, что схема наград даёт бонус за каждый корректно расположенный диск
    reward_scheme = Reward.from_config(settings)
    # 1) проверяем напрямую compute без среды
    rew, done = reward_scheme.compute(is_invalid=False, all_correct=False, correct_count=1)
    assert rew == settings.REWARD_STEP + settings.REWARD_CORRECT_PLACEMENT
    assert done is False

    # 2) убедимся, что среда сообщает корректное количество
    env = create_env(num_disks=3, num_sticks=3, max_steps=50, reward=reward_scheme)
    obs, info = env.reset()
    # вручную установим состояние, где один диск (самый большой) уже на последнем стержне
    env._state = [(2, 0), (0, 1), (0, 2)]
    obs, rew, terminated, truncated, info2 = env.step((0, 1))  # допустимый ход, без invalid
    # теперь диск0 остался на последнем стержне на правильной позиции (height0)
    assert info2.get("is_correct_placement", 0) == 1
    assert rew == settings.REWARD_STEP + settings.REWARD_CORRECT_PLACEMENT
