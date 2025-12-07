import mlflow
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os


class Callback:
    def on_episode_end(self, episode, logs):
        pass

    def on_train_end(self, logs):
        pass


class MLflowLogger(Callback):
    def __init__(self, experiment_name="RL_Experiment"):
        mlflow.set_experiment(experiment_name)
        # End all active runs (sometimes nested in notebooks)
        while mlflow.active_run() is not None:
            mlflow.end_run()
        self.run = mlflow.start_run()

    def on_episode_end(self, episode, logs):
        for k, v in logs.items():
            if k == "agent":
                continue
            if isinstance(v, (int, float, np.integer, np.floating)):
                mlflow.log_metric(k, float(v), step=episode)

    def on_train_end(self, logs):
        mlflow.end_run()


class TensorBoardLogger(Callback):
    def __init__(self, log_dir="runs"):
        self.writer = SummaryWriter(log_dir=log_dir)

    def on_episode_end(self, episode, logs):
        for k, v in logs.items():
            if k == "agent":
                continue
            if isinstance(v, (int, float, np.integer, np.floating)):
                self.writer.add_scalar(k, float(v), episode)

    def on_train_end(self, logs):
        self.writer.close()


class EarlyStopping(Callback):
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.counter = 0
        self.stopped = False

    def on_episode_end(self, episode, logs):
        reward = logs.get("total_reward", None)
        if reward is None:
            return
        if self.best is None or reward > self.best + self.min_delta:
            self.best = reward
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped = True


class CheckpointSaver(Callback):
    def __init__(self, save_dir="checkpoints"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def on_episode_end(self, episode, logs):
        agent = logs.get("agent", None)
        if agent is not None and hasattr(agent, "save"):
            agent.save(os.path.join(self.save_dir, f"agent_ep{episode}.pt"))
