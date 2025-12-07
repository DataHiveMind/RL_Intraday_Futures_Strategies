import numpy as np
import pandas as pd


class LimitOrderBook:
    """
    Simulate a simple limit order book (LOB) for a single asset.
    Supports order flow, bid-ask spread, impact, and microstructure noise.
    """

    def __init__(self, initial_bid, initial_ask, tick_size=0.01, depth=5, seed=None):
        self.tick_size = tick_size
        self.depth = depth
        self.rng = np.random.default_rng(seed)
        self.reset(initial_bid, initial_ask)

    def reset(self, initial_bid, initial_ask):
        self.bids = [(initial_bid - i * self.tick_size, 100) for i in range(self.depth)]
        self.asks = [(initial_ask + i * self.tick_size, 100) for i in range(self.depth)]
        self.last_trade = None

    def submit_limit_order(self, side, price, volume):
        book = self.bids if side == "buy" else self.asks
        for i, (p, v) in enumerate(book):
            if np.isclose(p, price):
                book[i] = (p, v + volume)
                return
        book.append((price, volume))
        book.sort(reverse=(side == "buy"))
        book[:] = book[: self.depth]

    def submit_market_order(self, side, volume):
        book = self.asks if side == "buy" else self.bids
        filled = 0
        trades = []
        for i, (p, v) in enumerate(book):
            if v == 0:
                continue
            fill = min(volume - filled, v)
            book[i] = (p, v - fill)
            trades.append((p, fill))
            filled += fill
            if filled >= volume:
                break
        self.last_trade = trades[-1][0] if trades else None
        return trades

    def get_top_of_book(self):
        best_bid = max(self.bids, key=lambda x: x[0]) if self.bids else (None, 0)
        best_ask = min(self.asks, key=lambda x: x[0]) if self.asks else (None, 0)
        return best_bid, best_ask

    def simulate_order_flow(self, n_steps=100, arrival_rate=0.5):
        """Simulate random order flow for both sides."""
        for _ in range(n_steps):
            if self.rng.random() < arrival_rate:
                # Randomly choose buy/sell, limit/market
                side = "buy" if self.rng.random() < 0.5 else "sell"
                is_market = self.rng.random() < 0.5
                volume = self.rng.integers(1, 10)
                if is_market:
                    self.submit_market_order(side, volume)
                else:
                    top_bid, top_ask = self.get_top_of_book()
                    price = top_bid[0] if side == "buy" else top_ask[0]
                    if price is not None:
                        price += self.tick_size * self.rng.integers(-1, 2)
                        self.submit_limit_order(side, price, volume)

    def get_book_snapshot(self):
        return {
            "bids": list(self.bids),
            "asks": list(self.asks),
            "last_trade": self.last_trade,
        }
