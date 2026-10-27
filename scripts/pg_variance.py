"""Week 6: measure how much a learned baseline reduces the variance of the REINFORCE gradient estimator.

For a fixed (partly trained) policy we sample N episodes and compute, per episode, the gradient estimate
    g          = sum_t  grad log pi(a_t|s_t) * G_t
    g_baseline = sum_t  grad log pi(a_t|s_t) * (G_t - V(s_t))
Both are unbiased estimates of the same policy gradient; we compare their mean squared deviation
tr Cov[g] = E||g - E[g]||^2 and the cosine similarity of their means.

Usage:  python scripts/pg_variance.py   ->   results/pg_variance.json
"""
import json
import os
import sys

import gymnasium as gym
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rl.common import ensure_dir
from rl.policy_gradient import discounted_returns, train_reinforce

OUT = ensure_dir(os.path.join(os.path.dirname(__file__), "..", "results"))


def gradient_samples(ac, env, n, gamma=0.99, seed=1000):
    params = list(ac.pi.parameters())
    plain, based = [], []
    for ep in range(n):
        s, _ = env.reset(seed=seed + ep)
        logps, values, rewards, done = [], [], [], False
        while not done:
            st = torch.as_tensor(s, dtype=torch.float32)
            d = ac.dist(st)
            a = d.sample()
            s, r, term, trunc, _ = env.step(int(a))
            done = term or trunc
            logps.append(d.log_prob(a)); rewards.append(r)
            with torch.no_grad():
                values.append(ac.v(st).item())
        G = torch.as_tensor(discounted_returns(rewards, gamma), dtype=torch.float32)
        V = torch.as_tensor(values, dtype=torch.float32)
        lp = torch.stack(logps)
        for weights, store in ((G, plain), (G - V, based)):
            grads = torch.autograd.grad((lp * weights).sum(), params, retain_graph=True)
            store.append(torch.cat([g.flatten() for g in grads]).numpy())
    return np.array(plain), np.array(based)


def main():
    results = {}
    for train_eps in (100, 300):  # an early policy and a better one
        ac, rets = train_reinforce("CartPole-v1", episodes=train_eps, seed=0)
        env = gym.make("CartPole-v1")
        g, gb = gradient_samples(ac, env, n=500)
        var_g = float(((g - g.mean(0)) ** 2).sum(1).mean())
        var_gb = float(((gb - gb.mean(0)) ** 2).sum(1).mean())
        cos = float(g.mean(0) @ gb.mean(0) / (np.linalg.norm(g.mean(0)) * np.linalg.norm(gb.mean(0))))
        results[f"policy after {train_eps} training episodes"] = {
            "policy_mean_return": round(float(np.mean(rets[-20:])), 1),
            "trace_cov_no_baseline": round(var_g, 1), "trace_cov_with_baseline": round(var_gb, 1),
            "variance_reduction_pct": round((1 - var_gb / var_g) * 100, 1),
            "cosine_similarity_of_mean_gradients": round(cos, 3),
        }
        print(train_eps, results[f"policy after {train_eps} training episodes"])
    with open(os.path.join(OUT, "pg_variance.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
