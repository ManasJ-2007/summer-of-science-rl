"""Q-learning vs SARSA on FrozenLake, Taxi and CliffWalking, with value iteration as the optimal reference.

Usage:  python scripts/run_tabular.py
Outputs: results/tabular_*.png and results/tabular_summary.csv
"""
import csv
import os
import sys

import gymnasium as gym
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rl.common import ensure_dir, latest_env, smooth
from rl.tabular import evaluate_greedy, td_control, value_iteration

OUT = ensure_dir(os.path.join(os.path.dirname(__file__), "..", "results"))
SEEDS = [0, 1, 2]

# (env id, kwargs, episodes, alpha, gamma, y-limits for the plot: early random episodes dwarf the rest)
ENVS = [
    (latest_env("FrozenLake"), dict(map_name="4x4", is_slippery=True), 20_000, 0.05, 0.99, None),
    (latest_env("Taxi"), {}, 5_000, 0.5, 0.99, (-200, 15)),
    (latest_env("CliffWalking"), {}, 500, 0.5, 1.0, (-150, 0)),
]


def main():
    rows = []
    for env_id, kw, episodes, alpha, gamma, ylim in ENVS:
        env = gym.make(env_id, **kw)
        Q_star, _, sweeps = value_iteration(env, gamma=gamma if gamma < 1 else 0.999)
        vi_score = evaluate_greedy(env, Q_star)
        rows.append([env_id, "value_iteration", "-", f"{vi_score:.3f}", "-", f"{sweeps} sweeps"])

        plt.figure(figsize=(7, 4))
        for method, color in (("qlearning", "tab:blue"), ("sarsa", "tab:orange")):
            curves, scores = [], []
            for seed in SEEDS:
                Q, rets = td_control(env, method, episodes, alpha=alpha, gamma=gamma, seed=seed)
                curves.append(smooth(rets, max(10, episodes // 100)))
                scores.append(evaluate_greedy(env, Q))
            curves = np.array(curves)
            x = np.arange(curves.shape[1])
            plt.plot(x, curves.mean(0), color=color, label=method)
            plt.fill_between(x, curves.min(0), curves.max(0), color=color, alpha=0.2)
            # last 10% of training episodes: includes exploration, unlike the greedy eval
            train_tail = float(np.mean([c[-max(1, episodes // 10):].mean() for c in curves]))
            rows.append([env_id, method, len(SEEDS), f"{np.mean(scores):.3f} ± {np.std(scores):.3f}",
                         f"{train_tail:.3f}", f"alpha={alpha}, gamma={gamma}, episodes={episodes}"])
            print(env_id, method, "greedy eval:", np.round(scores, 3))
        plt.axhline(vi_score, color="k", ls="--", lw=1, label="value-iteration policy")
        plt.title(f"{env_id}: training return (mean, min–max over {len(SEEDS)} seeds)")
        if ylim:
            plt.ylim(*ylim)
        plt.xlabel("episode"); plt.ylabel("smoothed return"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(OUT, f"tabular_{env_id.split('-')[0].lower()}.png"), dpi=130)
        plt.close()

    with open(os.path.join(OUT, "tabular_summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["env", "method", "seeds", "greedy_eval_return", "train_return_last10pct", "notes"])
        w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
