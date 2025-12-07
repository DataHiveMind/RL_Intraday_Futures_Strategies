"""Backtesting simulator for RL trading strategies."""

import numpy as np
import mlflow
from torch.utils.tensorboard import SummaryWriter


class BacktestSimulator:
    def __init__(self, env, agent, data, log_dir=None, mlflow_experiment=None):
        self.env = env
        self.agent = agent
        self.data = data
        self.writer = SummaryWriter(log_dir=log_dir) if log_dir else None
        self.mlflow_experiment = mlflow_experiment
        if mlflow_experiment:
            mlflow.set_experiment(mlflow_experiment)

    def run(self, n_episodes=1, epsilon=0.05):
        all_rewards = []
        for episode in range(n_episodes):
            state = self.env.reset()
            done = False
            total_reward = 0
            step = 0
            while not done:
                action = self.agent.select_action(state, epsilon=epsilon)
                next_state, reward, done, info = self.env.step(action)
                self.agent.store_transition((state, action, reward, next_state, done))
                loss = self.agent.update(step=step)
                state = next_state
                total_reward += reward
                if self.writer:
                    self.writer.add_scalar("Backtest/Reward", reward, step)
                    if loss is not None:
                        self.writer.add_scalar("Backtest/Loss", loss, step)
                step += 1
            self.agent.update_target()  # For DQN
            all_rewards.append(total_reward)
            if self.mlflow_experiment:
                mlflow.log_metric("backtest_total_reward", total_reward, step=episode)
            print(f"Episode {episode}: Total Reward = {total_reward}")
        return all_rewards
