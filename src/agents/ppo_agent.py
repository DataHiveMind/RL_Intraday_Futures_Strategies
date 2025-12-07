"""PPO Agent implementation using PyTorch (vectorized)."""

import torch
import torch.nn as nn
import torch.optim as optim


class PolicyNetwork(nn.Module):
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


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class PPOAgent:
    def __init__(
        self, state_dim, action_dim, lr=3e-4, gamma=0.99, clip_epsilon=0.2, device=None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.value_net = ValueNetwork(state_dim).to(self.device)
        self.optimizer = optim.Adam(
            list(self.policy_net.parameters()) + list(self.value_net.parameters()),
            lr=lr,
        )
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.memory = []

    def act(self, state, deterministic=False):
        """
        Select an action given a state. If deterministic, return the action with highest probability.
        Otherwise, sample from the policy distribution.
        """
        import torch
        import numpy as np

        self.policy_net.eval()
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32)
            if state_tensor.ndim == 1:
                state_tensor = state_tensor.unsqueeze(0)
            action_probs = self.policy_net(state_tensor)
            if hasattr(action_probs, "probs"):
                # If policy returns a distribution object
                if deterministic:
                    action = torch.argmax(action_probs.probs, dim=-1).item()
                else:
                    action = action_probs.sample().item()
            else:
                # Assume action_probs is a tensor of probabilities
                if deterministic:
                    action = torch.argmax(action_probs, dim=-1).item()
                else:
                    action_dist = torch.distributions.Categorical(logits=action_probs)
                    action = action_dist.sample().item()
        return action

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        logits = self.policy_net(state)
        probs = torch.softmax(logits, dim=-1)
        action = torch.multinomial(probs, 1).item()
        return action, probs[0][int(action)].item()

    def store_transition(self, transition):
        self.memory.append(transition)
        if len(self.memory) > 10000:
            self.memory.pop(0)

    def update(self):
        # Vectorized PPO update (placeholder, implement GAE, batching, etc. as needed)
        if len(self.memory) < 32:
            return
        # Unpack memory
        states, actions, rewards, next_states, dones, old_probs = zip(*self.memory)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        old_probs = torch.FloatTensor(old_probs).unsqueeze(1).to(self.device)

        # Compute values and advantages
        values = self.value_net(states)
        next_values = self.value_net(next_states)
        returns = rewards + self.gamma * next_values * (1 - dones)
        advantages = returns - values

        # Policy loss
        logits = self.policy_net(states)
        probs = torch.softmax(logits, dim=-1)
        action_probs = probs.gather(1, actions)
        ratio = action_probs / (old_probs + 1e-8)
        surr1 = ratio * advantages
        surr2 = (
            torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
            * advantages
        )
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = nn.functional.mse_loss(values, returns.detach())

        # Total loss
        loss = policy_loss + 0.5 * value_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.memory = []
        return loss.item()
