"""Slippage models for simulating transaction costs (vectorized)."""

import numpy as np


def simple_slippage(price, volume, rate=0.0005):
    """Apply a simple linear slippage model (vectorized)."""
    return price * (1 + rate * np.asarray(volume))


def square_root_slippage(price, volume, rate=0.0005):
    """Square-root impact model (vectorized)."""
    return price * (1 + rate * np.sqrt(np.abs(volume)))


def fixed_slippage(price, slippage=0.01):
    """Fixed slippage per trade (vectorized)."""
    return price + slippage


def no_slippage(price, volume):
    """No slippage (for idealized backtests)."""
    return price


# --- Additional slippage models ---
def exponential_slippage(price, volume, rate=0.0003):
    """Exponential impact model: slippage grows exponentially with volume."""
    return price * (1 + rate * (np.exp(np.abs(volume)) - 1))


def bid_ask_spread_slippage(price, spread=0.01, side="buy"):
    """Apply half-spread slippage depending on trade side."""
    if side == "buy":
        return price + spread / 2
    else:
        return price - spread / 2


def random_noise_slippage(price, std=0.002):
    """Add random Gaussian noise to simulate unpredictable slippage."""
    return price + np.random.normal(0, std, size=np.shape(price))


def volume_percentile_slippage(
    price, volume, percentiles=(0.25, 0.75), rates=(0.0002, 0.001)
):
    """Apply different slippage rates based on volume percentiles (vectorized)."""
    v = np.asarray(volume)
    p25, p75 = np.percentile(v, [percentiles[0] * 100, percentiles[1] * 100])
    rate = np.where(
        v < p25, rates[0], np.where(v > p75, rates[1], (rates[0] + rates[1]) / 2)
    )
    return price * (1 + rate * v)


def time_of_day_slippage(
    price, volume, time_index, open_rate=0.001, close_rate=0.0015, normal_rate=0.0005
):
    """Higher slippage at open/close, lower during normal hours. time_index: 0=open, 1=close, else normal."""
    rate = np.where(
        time_index == 0, open_rate, np.where(time_index == 1, close_rate, normal_rate)
    )
    return price * (1 + rate * np.asarray(volume))
