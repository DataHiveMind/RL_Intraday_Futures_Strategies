import numpy as np
import pandas as pd


def evaluate_agent(env, agent, n_episodes=10, render=False, callback=None):
    """Run evaluation episodes and collect results."""
    all_rewards = []
    all_balances = []
    all_actions = []
    all_infos = []
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        rewards = []
        balances = [info.get("balance", 0)]
        actions = []
        infos = [info]
        while not done:
            action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            rewards.append(reward)
            balances.append(info.get("balance", 0))
            actions.append(action)
            infos.append(info)
            if render:
                env.render()
            if callback:
                callback(locals())
        all_rewards.append(rewards)
        all_balances.append(balances)
        all_actions.append(actions)
        all_infos.append(infos)
    return {
        "rewards": all_rewards,
        "balances": all_balances,
        "actions": all_actions,
        "infos": all_infos,
    }


def aggregate_metrics(results):
    """Aggregate metrics from evaluation results."""
    rewards = [np.sum(r) for r in results["rewards"]]
    balances = [b[-1] for b in results["balances"]]
    actions = np.concatenate(results["actions"])
    metrics = {
        "mean_total_reward": np.mean(rewards),
        "std_total_reward": np.std(rewards),
        "mean_final_balance": np.mean(balances),
        "std_final_balance": np.std(balances),
        "action_distribution": np.bincount(actions) / len(actions),
    }
    return metrics


def evaluate_multiple_agents(env, agents, n_episodes=10):
    """Evaluate multiple agents and compare metrics."""
    results = {}
    for name, agent in agents.items():
        res = evaluate_agent(env, agent, n_episodes=n_episodes)
        metrics = aggregate_metrics(res)
        results[name] = metrics
    return pd.DataFrame(results)
