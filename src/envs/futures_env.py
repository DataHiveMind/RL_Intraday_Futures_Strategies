"""Custom RL environment for intraday futures trading (vectorized, Gym-compatible)."""

import gymnasium as gym
import numpy as np


class FuturesTradingEnv(gym.Env):
    def __init__(self, data, initial_balance=1e6, cost_fn=None, reward_fn=None):
        super().__init__()
        self.data = data
        self.initial_balance = initial_balance
        self.cost_fn = cost_fn
        self.reward_fn = reward_fn
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0  # +1=long, -1=short, 0=flat
        self.action_space = gym.spaces.Discrete(3)  # 0=hold, 1=buy, 2=sell
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(data.shape[1] + 3,), dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        obs = self._get_obs()
        info = {
            "balance": self.balance,
            "position": self.position,
            "step": self.current_step,
        }
        return obs, info

    def step(self, action):
        prev_balance = self.balance
        price = self.data.iloc[self.current_step]["close"]
        volume = 1  # For simplicity; replace with actual volume logic
        cost = self.cost_fn(price, volume) if self.cost_fn else 0
        reward = 0
        # Execute action
        if action == 1:  # Buy
            self.position = 1
            self.balance -= price + cost
        elif action == 2:  # Sell
            self.position = -1
            self.balance += price - cost
        # Reward
        if self.reward_fn:
            reward = self.reward_fn(self.balance, prev_balance)
        else:
            reward = self.balance - prev_balance
        self.current_step += 1
        terminated = self.current_step >= len(self.data) - 1
        truncated = False  # Add custom truncation logic if needed
        obs = self._get_obs()
        info = {
            "balance": self.balance,
            "position": self.position,
            "step": self.current_step,
        }
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        # Concatenate market data and agent state
        row = self.data.iloc[self.current_step].values
        obs = np.concatenate([row, [self.balance, self.position, self.current_step]])
        return obs.astype(np.float32)
