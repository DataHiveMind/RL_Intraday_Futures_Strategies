import mlflow
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

"""DQN Agent implementation using PyTorch."""


class DQNNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def save(self, path):
        """Save only the Q-network weights to avoid pickling errors."""
        torch.save(self.q_net.state_dict(), path)

    def act(self, state, epsilon=0.1, eval=False):
        """Select an action for the given state. If eval=True, use epsilon=0.0."""
        if eval:
            epsilon = 0.0
        return self.select_action(state, epsilon=epsilon)

    def __init__(
        self,
        state_dim,
        action_dim,
        lr=1e-3,
        gamma=0.99,
        device=None,
        log_dir=None,
        mlflow_experiment=None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.action_dim = action_dim
        self.memory = []  # Replace with a replay buffer for production
        self.batch_size = 32

        # MLflow and TensorBoard
        self.writer = SummaryWriter(log_dir=log_dir) if log_dir else None
        self.mlflow_experiment = mlflow_experiment
        if mlflow_experiment:
            mlflow.set_experiment(mlflow_experiment)

    def select_action(self, state, epsilon=0.1):
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state)
        return q_values.argmax().item()

    def store_transition(self, transition):
        self.memory.append(transition)
        if len(self.memory) > 10000:
            self.memory.pop(0)

    def update(self, step=None):
        import random

        if len(self.memory) < self.batch_size:
            return
        batch = list(zip(*random.sample(self.memory, self.batch_size)))
        states = torch.FloatTensor(np.vstack(batch[0])).to(self.device)
        actions = torch.LongTensor(batch[1]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(batch[2]).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.vstack(batch[3])).to(self.device)
        dones = torch.FloatTensor(batch[4]).unsqueeze(1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1, keepdim=True)[0]
            target = rewards + self.gamma * next_q * (1 - dones)
        loss = nn.functional.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # Log to TensorBoard
        if self.writer and step is not None:
            self.writer.add_scalar("Loss/DQN", loss.item(), step)
        # Log to MLflow
        if self.mlflow_experiment:
            mlflow.log_metric("dqn_loss", loss.item(), step=step)
        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())
