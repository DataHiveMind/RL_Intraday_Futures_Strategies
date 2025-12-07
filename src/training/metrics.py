import numpy as np


def episode_reward(rewards):
    return np.sum(rewards)


def average_reward(rewards):
    return np.mean(rewards)


def max_drawdown(balances):
    balances = np.array(balances)
    peak = np.maximum.accumulate(balances)
    drawdown = (peak - balances) / (peak + 1e-8)
    return np.max(drawdown)


def reward_std(rewards):
    return np.std(rewards)


def sharpe_ratio(rewards, risk_free_rate=0.0, eps=1e-8):
    mean_ret = np.mean(rewards)
    std_ret = np.std(rewards) + eps
    return (mean_ret - risk_free_rate) / std_ret


def custom_metrics(rewards, balances):
    return {
        "episode_reward": episode_reward(rewards),
        "average_reward": average_reward(rewards),
        "max_drawdown": max_drawdown(balances),
        "reward_std": reward_std(rewards),
        "sharpe_ratio": sharpe_ratio(rewards),
    }
