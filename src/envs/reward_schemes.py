import numpy as np


def pnl_reward(balance, prev_balance):
    """Reward is change in balance (PnL)."""
    return balance - prev_balance


def sharpe_reward(returns, risk_free_rate=0.0, eps=1e-8):
    """Reward is the Sharpe ratio of returns (vectorized)."""
    mean_ret = np.mean(returns)
    std_ret = np.std(returns) + eps
    return (mean_ret - risk_free_rate) / std_ret


def sortino_reward(returns, risk_free_rate=0.0, eps=1e-8):
    """Reward is the Sortino ratio of returns (vectorized)."""
    mean_ret = np.mean(returns)
    downside = np.std(np.minimum(returns - risk_free_rate, 0)) + eps
    return (mean_ret - risk_free_rate) / downside


def drawdown_penalty(balance_history, penalty=0.1):
    """Penalize large drawdowns in balance history."""
    peak = np.maximum.accumulate(balance_history)
    drawdown = (peak - balance_history) / (peak + 1e-8)
    return -penalty * np.max(drawdown)


def risk_adjusted_reward(balance, prev_balance, balance_history, penalty=0.1):
    """PnL minus drawdown penalty."""
    pnl = balance - prev_balance
    dd_penalty = drawdown_penalty(balance_history, penalty)
    return pnl + dd_penalty
