import numpy as np


def sharpe_ratio(returns, risk_free_rate=0.0, eps=1e-8):
    """Compute the Sharpe ratio."""
    mean_ret = np.mean(returns)
    std_ret = np.std(returns) + eps
    return (mean_ret - risk_free_rate) / std_ret


def sortino_ratio(returns, risk_free_rate=0.0, eps=1e-8):
    """Compute the Sortino ratio."""
    mean_ret = np.mean(returns)
    downside = np.std(np.minimum(returns - risk_free_rate, 0)) + eps
    return (mean_ret - risk_free_rate) / downside


def max_drawdown(balance):
    """Compute the maximum drawdown."""
    balance = np.array(balance)
    peak = np.maximum.accumulate(balance)
    drawdown = (peak - balance) / (peak + 1e-8)
    return np.max(drawdown)


def calmar_ratio(returns, balance, eps=1e-8):
    """Compute the Calmar ratio: mean return / max drawdown."""
    mdd = max_drawdown(balance)
    mean_ret = np.mean(returns)
    return mean_ret / (mdd + eps)


def volatility(returns):
    """Annualized volatility (assuming daily returns)."""
    return np.std(returns) * np.sqrt(252)


def omega_ratio(returns, threshold=0.0, eps=1e-8):
    """Omega ratio: ratio of gains above threshold to losses below threshold."""
    returns = np.array(returns)
    gains = returns[returns > threshold] - threshold
    losses = threshold - returns[returns < threshold]
    return (np.sum(gains) + eps) / (np.sum(losses) + eps)


def tail_ratio(returns, eps=1e-8):
    """Tail ratio: ratio of 95th to 5th percentile returns."""
    p95 = np.percentile(returns, 95)
    p5 = np.percentile(returns, 5) + eps
    return p95 / p5


def skewness(returns):
    """Sample skewness of returns."""
    returns = np.array(returns)
    mean = np.mean(returns)
    std = np.std(returns) + 1e-8
    return np.mean(((returns - mean) / std) ** 3)


def kurtosis(returns):
    """Sample kurtosis of returns."""
    returns = np.array(returns)
    mean = np.mean(returns)
    std = np.std(returns) + 1e-8
    return np.mean(((returns - mean) / std) ** 4)
