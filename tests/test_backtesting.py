def test_placeholder():
import numpy as np
import pytest
from types import SimpleNamespace
from src.backtesting.simulator import BacktestSimulator

class DummyEnv:
    def reset(self):
        return np.zeros(4)
    def step(self, action):
        return np.zeros(4), 1.0, True, {}

class DummyAgent:
    def __init__(self):
        self.transitions = []
    def select_action(self, state, epsilon=0.1):
        return 0
    def store_transition(self, transition):
        self.transitions.append(transition)
    def update(self, step=0):
        return 0.0
    def update_target(self):
        pass

def test_backtest_simulator_run():
    env = DummyEnv()
    agent = DummyAgent()
    data = np.zeros((10, 4))
    sim = BacktestSimulator(env, agent, data)
    rewards = sim.run(n_episodes=2, epsilon=0.0)
    assert len(rewards) == 2
