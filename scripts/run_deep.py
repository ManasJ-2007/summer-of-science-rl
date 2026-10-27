"""Deep RL experiments: algorithm comparison on CartPole, DQN ablation, DDPG on Pendulum, PPO on LunarLander.

Usage:  python scripts/run_deep.py [cartpole|ablation|pendulum|lunar|all]
Runs seeds in parallel processes. Outputs: results/<experiment>.png, results/deep_summary.csv, results/raw/*.json
"""
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import gymnasium as gym
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rl.common import ensure_dir, smooth
from rl.ddpg import train_ddpg
from rl.dqn import train_dqn
from rl.policy_gradient import train_a2c, train_ppo, train_reinforce

OUT = ensure_dir(os.path.join(os.path.dirname(__file__), "..", "results"))
RAW = ensure_dir(os.path.join(OUT, "raw"))
SEEDS = [0, 1, 2]

# name -> (env, trainer, kwargs, kind)  kind decides how the trained policy acts at evaluation time
EXPERIMENTS = {
    "cartpole": {
        "DQN": ("CartPole-v1", train_dqn, dict(total_steps=60_000), "q"),
        "REINFORCE (baseline)": ("CartPole-v1", train_reinforce, dict(episodes=1_000), "pi"),
        "A2C": ("CartPole-v1", train_a2c, dict(total_steps=150_000), "pi"),
        "PPO": ("CartPole-v1", train_ppo, dict(total_steps=150_000), "pi"),
    },
    "ablation": {
        "DQN (replay + target)": ("CartPole-v1", train_dqn, dict(total_steps=60_000), "q"),
        "Double DQN": ("CartPole-v1", train_dqn, dict(total_steps=60_000, double=True), "q"),
        "no target network": ("CartPole-v1", train_dqn, dict(total_steps=60_000, use_target=False), "q"),
        "no replay buffer": ("CartPole-v1", train_dqn, dict(total_steps=60_000, use_replay=False), "q"),
    },
    "pendulum": {
        "DDPG": ("Pendulum-v1", train_ddpg, dict(total_steps=30_000), "det"),
    },
    "lunar": {
        "PPO": ("LunarLander-v3", train_ppo, dict(total_steps=1_000_000, ent_coef=0.01), "pi"),
    },
    # Week 6: does the learned baseline matter, and how do REINFORCE / A2C compare with PPO on LunarLander?
    "reinforce": {
        "REINFORCE (baseline)": ("CartPole-v1", train_reinforce, dict(episodes=1_000), "pi"),
        "REINFORCE (no baseline)": ("CartPole-v1", train_reinforce, dict(episodes=1_000, baseline=False), "pi"),
    },
    "lunar_pg": {
        "REINFORCE (baseline)": ("LunarLander-v3", train_reinforce, dict(episodes=3_000), "pi"),
        "A2C": ("LunarLander-v3", train_a2c, dict(total_steps=1_000_000), "pi"),
    },
    # Week 5: one-at-a-time sensitivity around the default DQN (eps_frac=0.2, batch=64, target_update=500)
    "dqn_tuning": {
        "default": ("CartPole-v1", train_dqn, dict(total_steps=60_000), "q"),
        "ε-decay 10% of steps": ("CartPole-v1", train_dqn, dict(total_steps=60_000, eps_frac=0.1), "q"),
        "ε-decay 50% of steps": ("CartPole-v1", train_dqn, dict(total_steps=60_000, eps_frac=0.5), "q"),
        "batch 32": ("CartPole-v1", train_dqn, dict(total_steps=60_000, batch_size=32), "q"),
        "batch 128": ("CartPole-v1", train_dqn, dict(total_steps=60_000, batch_size=128), "q"),
        "target update 100": ("CartPole-v1", train_dqn, dict(total_steps=60_000, target_update=100), "q"),
        "target update 2000": ("CartPole-v1", train_dqn, dict(total_steps=60_000, target_update=2000), "q"),
    },
}


def act(model, kind, s, env):
    st = torch.as_tensor(s, dtype=torch.float32)
    with torch.no_grad():
        if kind == "q":
            return int(model(st).argmax())
        if kind == "pi":
            return int(model.pi(st).argmax())  # greedy action of the stochastic policy
        return model(st).numpy() * float(env.action_space.high[0])


def evaluate(model, kind, env_id, episodes=20, seed=50_000):
    env = gym.make(env_id)
    rets = []
    for ep in range(episodes):
        s, _ = env.reset(seed=seed + ep)
        done, G = False, 0.0
        while not done:
            s, r, term, trunc, _ = env.step(act(model, kind, s, env))
            done, G = term or trunc, G + r
        rets.append(G)
    env.close()
    return float(np.mean(rets))


def job(args):
    exp, name, seed = args
    torch.set_num_threads(1)
    env_id, fn, kw, kind = EXPERIMENTS[exp][name]
    model, returns = fn(env_id=env_id, seed=seed, **kw)
    score = evaluate(model, kind, env_id)
    return exp, name, seed, [float(x) for x in returns], score


def plot(exp, results):
    plt.figure(figsize=(7.5, 4.2))
    for name, runs in results.items():
        n = min(len(r) for r, _ in runs)
        curves = np.array([smooth(r[:n], 20) for r, _ in runs])
        x = np.arange(n)
        line, = plt.plot(x, curves.mean(0), label=name)
        plt.fill_between(x, curves.min(0), curves.max(0), color=line.get_color(), alpha=0.15)
    env_id = next(iter(EXPERIMENTS[exp].values()))[0]
    plt.title(f"{env_id} – {exp}: training return (mean, min–max over {len(SEEDS)} seeds)")
    plt.xlabel("episode"); plt.ylabel("return (smoothed, window 20)"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUT, f"deep_{exp}.png"), dpi=130); plt.close()


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    exps = list(EXPERIMENTS) if which == "all" else [which]
    jobs = [(e, n, s) for e in exps for n in EXPERIMENTS[e] for s in SEEDS]
    results = {e: {n: [] for n in EXPERIMENTS[e]} for e in exps}
    with ProcessPoolExecutor(max_workers=min(len(jobs), max(1, (os.cpu_count() or 2) - 2))) as pool:
        for exp, name, seed, rets, score in pool.map(job, jobs):
            results[exp][name].append((rets, score))
            print(f"{exp:9s} {name:22s} seed={seed} episodes={len(rets):5d} "
                  f"last20={np.mean(rets[-20:]):8.1f} greedy_eval={score:8.1f}", flush=True)

    summary = os.path.join(OUT, "deep_summary.csv")
    old = []
    if os.path.exists(summary):
        with open(summary, encoding="utf-8") as f:
            old = [r for r in csv.reader(f)][1:]
        old = [r for r in old if r[0] not in exps]
    rows = old
    for exp in exps:
        with open(os.path.join(RAW, f"{exp}.json"), "w") as f:
            json.dump({n: [{"returns": r, "greedy_eval": s} for r, s in v] for n, v in results[exp].items()}, f)
        plot(exp, results[exp])
        for name, runs in results[exp].items():
            last = [np.mean(r[-20:]) for r, _ in runs]
            ev = [s for _, s in runs]
            rows.append([exp, name, next(iter(EXPERIMENTS[exp].values()))[0], len(runs),
                         f"{np.mean(last):.1f} ± {np.std(last):.1f}", f"{np.mean(ev):.1f} ± {np.std(ev):.1f}"])
    with open(summary, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "algorithm", "env", "seeds", "train_return_last20", "greedy_eval_return_20ep"])
        w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
