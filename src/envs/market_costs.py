import numpy as np


def linear_cost(price, volume, rate=0.0005):
    """Linear transaction cost: cost = rate * price * volume"""
    return rate * np.abs(price * volume)


def quadratic_cost(price, volume, rate=0.0005, alpha=0.1):
    """Quadratic cost: cost = rate * price * volume + alpha * (volume ** 2)"""
    return rate * np.abs(price * volume) + alpha * (volume**2)


def fixed_cost(price, volume, fixed_fee=1.0):
    """Fixed cost per transaction, regardless of size."""
    return np.where(np.abs(volume) > 0, fixed_fee, 0.0)


def bid_ask_spread_cost(price, volume, spread=0.01):
    """Cost based on bid-ask spread: cost = 0.5 * spread * price * |volume|"""
    return 0.5 * spread * np.abs(price * volume)


def impact_cost(price, volume, impact_coeff=0.0001):
    """Market impact cost: cost = impact_coeff * (volume ** 2) * price"""
    return impact_coeff * (volume**2) * price


def total_cost(price, volume, cost_fns=None):
    """Aggregate multiple cost functions (vectorized)."""
    if cost_fns is None:
        cost_fns = [linear_cost]
    total = np.zeros_like(price, dtype=float)
    for fn in cost_fns:
        total += fn(price, volume)
    return total
