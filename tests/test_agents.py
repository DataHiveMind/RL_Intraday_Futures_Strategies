import numpy as np
import torch
from src.agents.dqn_agent import DQNAgent, DQNNetwork


def test_dqn_network_forward():
    state_dim, action_dim = 4, 3
    net = DQNNetwork(state_dim, action_dim)
    x = torch.randn(2, state_dim)
    out = net(x)
    assert out.shape == (2, action_dim)


def test_dqn_agent_select_action():
    state_dim, action_dim = 4, 3
    agent = DQNAgent(state_dim, action_dim)
    state = np.random.randn(state_dim)
    action = agent.select_action(state, epsilon=0.0)
    assert 0 <= action < action_dim
