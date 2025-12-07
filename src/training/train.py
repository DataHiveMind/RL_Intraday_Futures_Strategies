import numpy as np
from .metrics import custom_metrics


def train_agent(
    agent, env, n_episodes=100, callbacks=None, eval_env=None, eval_interval=10
):
    if callbacks is None:
        callbacks = []
    logs = {}
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        rewards = []
        balances = [info.get("balance", 0)]
        actions = []
        while not done:
            action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            rewards.append(reward)
            balances.append(info.get("balance", 0))
            actions.append(action)
        metrics = custom_metrics(rewards, balances)
        logs = {**metrics, "episode": episode, "agent": agent}
        for cb in callbacks:
            cb.on_episode_end(episode, logs)
        # Early stopping
        for cb in callbacks:
            if hasattr(cb, "stopped") and cb.stopped:
                print(f"Early stopping at episode {episode}")
                for cb2 in callbacks:
                    cb2.on_train_end(logs)
                return
        # Periodic evaluation
        if eval_env is not None and eval_interval and episode % eval_interval == 0:
            eval_obs, eval_info = eval_env.reset()
            eval_done = False
            eval_rewards = []
            while not eval_done:
                eval_action = agent.act(eval_obs, eval=True)
                eval_obs, eval_reward, eval_terminated, eval_truncated, eval_info = (
                    eval_env.step(eval_action)
                )
                eval_done = eval_terminated or eval_truncated
                eval_rewards.append(eval_reward)
            print(f"Eval episode {episode}: reward={np.sum(eval_rewards):.2f}")
    for cb in callbacks:
        cb.on_train_end(logs)
