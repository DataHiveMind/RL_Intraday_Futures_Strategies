import numpy as np
from src.training.metrics import (
    episode_reward,
    average_reward,
    max_drawdown,
    reward_std,
    sharpe_ratio,
    custom_metrics,
)


def test_episode_reward():
    rewards = [1, 2, 3]
    assert episode_reward(rewards) == 6


def test_average_reward():
    rewards = [1, 2, 3]
    assert average_reward(rewards) == 2


def test_max_drawdown():
    balances = [100, 120, 110, 90, 130]
    dd = max_drawdown(balances)
    assert 0 <= dd <= 1


def test_reward_std():
    rewards = [1, 2, 3]
    assert np.isclose(reward_std(rewards), np.std(rewards))


def test_sharpe_ratio():
    rewards = [1, 2, 3]
    sr = sharpe_ratio(rewards)
    assert isinstance(sr, float)


def test_custom_metrics():
    rewards = [1, 2, 3]
    balances = [100, 120, 110, 90, 130]
    metrics = custom_metrics(rewards, balances)
    assert "episode_reward" in metrics
