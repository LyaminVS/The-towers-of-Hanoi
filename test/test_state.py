import pytest
import torch

from env import state


def test_get_initial_state_basic():
    s = state.get_initial_state(3)
    assert s == ((0, 0), (0, 1), (0, 2))


@pytest.mark.parametrize("n", [1, 2, 5])
def test_get_initial_state_structure(n):
    s = state.get_initial_state(n)

    assert isinstance(s, tuple)
    assert len(s) == n

    for i, (stick, height) in enumerate(s):
        assert stick == 0
        assert height == i


def test_get_initial_state_invalid():
    with pytest.raises(ValueError):
        state.get_initial_state(0)


def test_state_to_observation():
    s = ((1, 0), (2, 1), (0, 2))
    obs = state.state_to_observation(s)

    assert isinstance(obs, torch.Tensor)
    assert obs.dtype == torch.float32
    assert obs.tolist() == [1.0, 0.0, 2.0, 1.0, 0.0, 2.0]


def test_observation_to_state_from_tensor():
    obs = torch.tensor([2.0, 0.0, 2.0, 1.0, 2.0, 2.0])
    s = state.observation_to_state(obs)

    assert s == ((2, 0), (2, 1), (2, 2))


def test_observation_to_state_invalid_length():
    with pytest.raises(ValueError):
        state.observation_to_state([0, 0, 1])


def test_roundtrip():
    s0 = state.get_initial_state(4)
    obs = state.state_to_observation(s0)
    s1 = state.observation_to_state(obs)

    assert s0 == s1


def test_is_terminal_state_true():
    s = ((2, 0), (2, 1), (2, 2))
    assert state.is_terminal_state(s, 3) is True


def test_is_terminal_state_false_wrong_stick():
    s = ((2, 0), (1, 1), (2, 2))
    assert state.is_terminal_state(s, 3) is False


def test_is_terminal_state_false_wrong_height():
    s = ((2, 0), (2, 2), (2, 1))
    assert state.is_terminal_state(s, 3) is False


def test_is_terminal_state_length_mismatch():
    s = ((2, 0), (2, 1), (2, 2))
    with pytest.raises(ValueError):
        state.is_terminal_state(s, 4)