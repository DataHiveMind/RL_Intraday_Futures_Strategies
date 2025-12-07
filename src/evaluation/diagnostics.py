import numpy as np
import pandas as pd


def log_episode(trades, rewards, balances, actions, info=None):
    """Log episode-level diagnostics as a DataFrame."""
    df = pd.DataFrame(
        {"trade": trades, "reward": rewards, "balance": balances, "action": actions}
    )
    if info is not None:
        for k, v in info.items():
            df[k] = v
    return df


def detect_anomalies(series, z_thresh=3.0):
    """Detect outliers in a series using z-score."""
    z = (series - np.mean(series)) / (np.std(series) + 1e-8)
    return np.where(np.abs(z) > z_thresh)[0]


def trade_summary(trades):
    """Summarize trade statistics: count, win rate, avg PnL, max/min PnL."""
    trades = np.array(trades)
    summary = {
        "count": len(trades),
        "win_rate": np.mean(trades > 0) if len(trades) > 0 else 0.0,
        "avg_pnl": np.mean(trades) if len(trades) > 0 else 0.0,
        "max_pnl": np.max(trades) if len(trades) > 0 else 0.0,
        "min_pnl": np.min(trades) if len(trades) > 0 else 0.0,
        "std_pnl": np.std(trades) if len(trades) > 0 else 0.0,
    }
    return summary


def action_distribution(actions, n_actions=None):
    """Return the distribution of actions taken."""
    actions = np.array(actions)
    if n_actions is None:
        n_actions = int(actions.max()) + 1
    counts = np.bincount(actions, minlength=n_actions)
    return counts / counts.sum()


def state_coverage(states, bins=10):
    """Estimate state space coverage using histogram binning."""
    states = np.array(states)
    if states.ndim == 1:
        states = states[:, None]
    coverage = [
        np.histogram(states[:, i], bins=bins)[0] for i in range(states.shape[1])
    ]
    return np.array(coverage)


def episode_length(rewards):
    """Return the length of an episode."""
    return len(rewards)


def rolling_sharpe(returns, window=50, risk_free_rate=0.0):
    """Compute rolling Sharpe ratio for a series of returns."""
    returns = np.array(returns)
    if len(returns) < window:
        return np.array([])
    roll_mean = pd.Series(returns).rolling(window).mean()
    roll_std = pd.Series(returns).rolling(window).std() + 1e-8
    sharpe = (roll_mean - risk_free_rate) / roll_std
    return sharpe.values


def detect_regime_change(series, window=100, threshold=2.0):
    """Detect regime changes using rolling mean/variance shifts."""
    s = pd.Series(series)
    roll_mean = s.rolling(window).mean()
    roll_std = s.rolling(window).std()
    diffs = np.abs(roll_mean.diff()) / (roll_std + 1e-8)
    return np.where(diffs > threshold)[0]
