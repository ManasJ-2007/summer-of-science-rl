import os
import random

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def mlp(in_dim, out_dim, hidden=(128, 128), act=nn.ReLU, out_act=None):
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), act()]
        d = h
    layers.append(nn.Linear(d, out_dim))
    if out_act is not None:
        layers.append(out_act())
    return nn.Sequential(*layers)


def smooth(x, w=20):
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        return x
    c = np.cumsum(np.insert(x, 0, 0.0))
    return np.concatenate([x[: w - 1].cumsum() / np.arange(1, w), (c[w:] - c[:-w]) / w])


def latest_env(name):
    """Newest registered version of an environment, e.g. 'Taxi' -> 'Taxi-v4' (versions differ across gymnasium releases)."""
    import gymnasium as gym
    versions = [int(k.rsplit("-v", 1)[1]) for k in gym.registry if k.rsplit("-v", 1)[0] == name]
    return f"{name}-v{max(versions)}"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path
