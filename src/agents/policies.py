"""Policy utilities for RL agents (vectorized)."""

import numpy as np
import torch


def epsilon_greedy(q_values, epsilon):
    """Vectorized epsilon-greedy for batch Q-values (PyTorch or numpy)."""
    if isinstance(q_values, torch.Tensor):
        q_values = q_values.detach().cpu().numpy()
    batch_size = q_values.shape[0] if q_values.ndim > 1 else 1
    actions = np.argmax(q_values, axis=-1) if batch_size > 1 else np.argmax(q_values)
    if batch_size == 1:
        if np.random.rand() < epsilon:
            return np.random.randint(q_values.shape[-1])
        return actions
    mask = np.random.rand(batch_size) < epsilon
    random_actions = np.random.randint(q_values.shape[-1], size=batch_size)
    actions = np.where(mask, random_actions, actions)
    return actions


def softmax_policy(q_values, tau=1.0):
    """Vectorized softmax policy for batch Q-values (PyTorch or numpy)."""
    if isinstance(q_values, torch.Tensor):
        q_values = q_values.detach().cpu().numpy()
    q = (q_values - np.max(q_values, axis=-1, keepdims=True)) / tau
    exp_q = np.exp(q)
    probs = exp_q / np.sum(exp_q, axis=-1, keepdims=True)
    if probs.ndim == 1:
        return np.random.choice(len(probs), p=probs)
    return np.array([np.random.choice(len(p), p=p) for p in probs])
