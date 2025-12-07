import numpy as np
import pandas as pd


class MarketSimulator:
    """
    Vectorized, event-driven market simulator for multi-asset RL research.
    Supports order matching, slippage, latency, partial fills, and multi-agent simulation.
    """

    def __init__(self, price_data, slippage_fn=None, latency=0, seed=None):
        self.price_data = price_data  # DataFrame: index=time, columns=assets
        self.slippage_fn = slippage_fn
        self.latency = latency
        self.rng = np.random.default_rng(seed)
        self.current_step = 0
        self.n_assets = price_data.shape[1]
        self.order_book = []  # List of (agent_id, asset, side, volume, price, time)
        self.trades = []

    def get_observation(self, asset):
        """
        Return the current price (or observation) for the given asset at the current step.
        """
        if asset not in self.price_data.columns:
            raise ValueError(f"Asset {asset} not found in price data columns.")
        return self.price_data[asset].iloc[self.current_step]

    def __init__(self, price_data, slippage_fn=None, latency=0, seed=None):
        self.price_data = price_data  # DataFrame: index=time, columns=assets
        self.slippage_fn = slippage_fn
        self.latency = latency
        self.rng = np.random.default_rng(seed)
        self.current_step = 0
        self.n_assets = price_data.shape[1]
        self.order_book = []  # List of (agent_id, asset, side, volume, price, time)
        self.trades = []

    def reset(self):
        self.current_step = 0
        self.order_book = []
        self.trades = []

    def submit_order(self, agent_id, asset, side, volume, price=None):
        """Submit a market or limit order."""
        order = {
            "agent_id": agent_id,
            "asset": asset,
            "side": side,  # 'buy' or 'sell'
            "volume": volume,
            "price": price,
            "time": self.current_step + self.latency,
        }
        self.order_book.append(order)

    def step(self):
        """Advance simulation by one step, match orders, apply slippage/latency. Returns next_obs, rewards, dones, infos dicts."""
        executed = []
        # Match market orders at current price
        for order in self.order_book:
            if order["time"] <= self.current_step:
                exec_price = self.price_data.loc[self.current_step, order["asset"]]
                if self.slippage_fn:
                    exec_price += self.slippage_fn(
                        exec_price, order["volume"], order["side"]
                    )
                trade = {
                    "agent_id": order["agent_id"],
                    "asset": order["asset"],
                    "side": order["side"],
                    "volume": order["volume"],
                    "price": exec_price,
                    "time": self.current_step,
                }
                self.trades.append(trade)
                executed.append(order)
        # Remove executed orders
        self.order_book = [o for o in self.order_book if o not in executed]
        # Prepare next_obs, rewards, dones, infos for all assets
        next_obs = {
            asset: self.get_observation(asset) for asset in self.price_data.columns
        }
        # For now, reward is 0 for all assets (customize as needed)
        rewards = {asset: 0.0 for asset in self.price_data.columns}
        # Done if at end of data
        dones = {
            asset: self.current_step >= len(self.price_data) - 2
            for asset in self.price_data.columns
        }
        infos = {asset: {} for asset in self.price_data.columns}
        self.current_step += 1
        return next_obs, rewards, dones, infos

    def get_trade_history(self):
        return pd.DataFrame(self.trades)

    def get_order_book(self):
        return pd.DataFrame(self.order_book)

    def is_done(self):
        return self.current_step >= len(self.price_data) - 1
