"""Weeks 2, 3 and 7: bandits, Bellman equations, policy vs value iteration, TD vs MC, RLHF and MARL.

Usage:  python scripts/run_foundations.py        (about a minute on a CPU)
Outputs: results/foundations_*.png and results/foundations_summary.json
"""
import json
import os
import sys
import time

import gymnasium as gym
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rl import dp, marl, rlhf, td
from rl.bandits import run_testbed
from rl.common import ensure_dir
from rl.tabular import evaluate_greedy, value_iteration

OUT = ensure_dir(os.path.join(os.path.dirname(__file__), "..", "results"))
summary = {}


def save(name):
    plt.tight_layout(); plt.savefig(os.path.join(OUT, f"foundations_{name}.png"), dpi=130); plt.close()


# ------------------------------------------------------------------ Week 2: bandits
def week2_bandits():
    configs = {"greedy (ε=0)": dict(eps=0.0), "ε=0.01": dict(eps=0.01), "ε=0.1": dict(eps=0.1),
               "optimistic Q₀=5, ε=0": dict(eps=0.0, q0=5.0), "UCB c=2": dict(ucb_c=2.0)}
    res = {}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name, kw in configs.items():
        r, opt = run_testbed(**kw)
        res[name] = {"avg_reward_last100": round(float(r[-100:].mean()), 3),
                     "pct_optimal_last100": round(float(opt[-100:].mean() * 100), 1)}
        ax[0].plot(r, label=name, lw=1); ax[1].plot(opt * 100, label=name, lw=1)
    ax[0].set(xlabel="step", ylabel="average reward", title="10-armed testbed (2,000 runs)")
    ax[1].set(xlabel="step", ylabel="% optimal action"); ax[1].legend(fontsize=8)
    save("bandits")
    summary["week2_bandits"] = res


# ------------------------------------------------------------------ Week 2: Bellman expectation equation
def week2_bellman():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    P, R = dp.mdp_matrices(env)
    gamma = 0.9
    pi = np.full(R.shape, 1 / R.shape[1])  # uniform random policy
    v_exact = dp.policy_evaluation_exact(P, R, pi, gamma)
    v_iter, deltas = dp.policy_evaluation_iterative(P, R, pi, gamma, tol=1e-12)
    ratios = np.array(deltas[1:]) / np.array(deltas[:-1])
    plt.figure(figsize=(6, 4))
    plt.semilogy(deltas, label="‖v_{k+1} − v_k‖∞")
    plt.semilogy(deltas[0] * gamma ** np.arange(len(deltas)), "--", label="γᵏ bound (γ = 0.9)")
    plt.xlabel("sweep k"); plt.ylabel("sup-norm change"); plt.legend()
    plt.title("Iterative policy evaluation is a γ-contraction")
    save("bellman_contraction")
    summary["week2_bellman"] = {
        "env": "FrozenLake 4x4 (slippery), uniform random policy, gamma=0.9",
        "max_abs_diff_exact_vs_iterative": float(np.max(np.abs(v_exact - v_iter))),
        "sweeps_to_1e-12": len(deltas),
        "max_observed_contraction_ratio": round(float(ratios.max()), 4),
        "v_start_state": round(float(v_exact[0]), 5),
    }


# ------------------------------------------------------------------ Week 3: policy vs value iteration
def week3_dp():
    out = {}
    for name, kw in (("FrozenLake 4x4", dict(map_name="4x4")), ("FrozenLake 8x8", dict(map_name="8x8"))):
        env = gym.make("FrozenLake-v1", is_slippery=True, **kw)
        P, R = dp.mdp_matrices(env)
        gamma = 0.99
        t = time.perf_counter(); pi_pi, v_pi, n_impr = dp.policy_iteration(P, R, gamma); t_pi = time.perf_counter() - t
        t = time.perf_counter(); Q_vi, v_vi, sweeps = value_iteration(env, gamma=gamma, tol=1e-10); t_vi = time.perf_counter() - t
        pi_vi = np.zeros_like(pi_pi); pi_vi[np.arange(len(v_vi)), Q_vi.argmax(1)] = 1
        out[name] = {
            "policy_iteration_improvement_steps": n_impr, "value_iteration_sweeps": sweeps,
            "max_abs_diff_v": float(np.max(np.abs(v_pi - v_vi))),
            "same_greedy_value": bool(np.allclose(v_pi, dp.policy_evaluation_exact(P, R, pi_vi, gamma), atol=1e-6)),
            "optimal_success_rate_eval_1000ep": round(evaluate_greedy(env, Q_vi), 3),
            "time_ms_policy_iteration": round(t_pi * 1000, 1), "time_ms_value_iteration": round(t_vi * 1000, 1),
        }
        if name == "FrozenLake 4x4":
            arrows = np.array(["←", "↓", "→", "↑"])[pi_pi.argmax(1)].reshape(4, 4)
            desc = env.unwrapped.desc.astype(str)
            fig, ax = plt.subplots(1, 2, figsize=(9, 4))
            im = ax[0].imshow(v_pi.reshape(4, 4), cmap="viridis")
            fig.colorbar(im, ax=ax[0]); ax[0].set_title("Optimal state values v* (γ=0.99)")
            ax[1].imshow(np.zeros((4, 4)), cmap="Greys", vmin=0, vmax=1)
            for r in range(4):
                for c in range(4):
                    label = desc[r, c] if desc[r, c] in "HG" else arrows[r, c]
                    ax[0].text(c, r, f"{v_pi[r * 4 + c]:.2f}", ha="center", va="center", color="w", fontsize=8)
                    ax[1].text(c, r, label, ha="center", va="center", fontsize=16,
                               color="crimson" if desc[r, c] == "H" else "k")
            ax[1].set_title("Optimal policy (H = hole, G = goal)")
            for a in ax: a.set_xticks([]); a.set_yticks([])
            save("frozenlake_policy")
    summary["week3_dp"] = out


# ------------------------------------------------------------------ Week 3: TD(0) vs Monte Carlo
def week3_td():
    plt.figure(figsize=(6.5, 4))
    res = {}
    for method, alphas, ls in (("td", (0.05, 0.1, 0.15), "-"), ("mc", (0.01, 0.02, 0.04), "--")):
        for a in alphas:
            curve = td.rms_curve(method, a)
            res[f"{method.upper()} α={a}"] = round(float(curve[-1]), 4)
            plt.plot(curve, ls, label=f"{method.upper()} α={a}")
    plt.xlabel("episodes"); plt.ylabel("RMS error (100 runs)"); plt.legend(fontsize=8)
    plt.title("Random walk: TD(0) vs constant-α MC")
    save("td_vs_mc")
    best_td = min(v for k, v in res.items() if k.startswith("TD"))
    best_mc = min(v for k, v in res.items() if k.startswith("MC"))
    summary["week3_td_vs_mc"] = {"rms_after_100_episodes": res, "best_td": best_td, "best_mc": best_mc,
                                 "td_error_reduction_pct": round((1 - best_td / best_mc) * 100, 1)}


# ------------------------------------------------------------------ Week 7: RLHF
def week7_rlhf():
    rng = np.random.default_rng(0)
    k = 20
    r_true = rng.normal(0, 1, k)
    out = {}
    ns = [50, 100, 250, 500, 1000, 2500, 5000]
    pair_acc, top_ok = [], []
    for n in ns:
        accs, tops = [], []
        for trial in range(20):
            w, l = rlhf.simulate_preferences(r_true, n, np.random.default_rng(trial))
            r_hat = rlhf.fit_bradley_terry(w, l, k)
            i, j = np.triu_indices(k, 1)
            accs.append(np.mean(np.sign(r_hat[i] - r_hat[j]) == np.sign(r_true[i] - r_true[j])))
            tops.append(r_hat.argmax() == r_true.argmax())
        pair_acc.append(float(np.mean(accs))); top_ok.append(float(np.mean(tops)))
    out["pairwise_order_accuracy"] = {str(n): round(a, 3) for n, a in zip(ns, pair_acc)}
    out["best_response_identified_rate"] = {str(n): round(t, 2) for n, t in zip(ns, top_ok)}

    w, l = rlhf.simulate_preferences(r_true, 2500, np.random.default_rng(99))
    r_hat = rlhf.fit_bradley_terry(w, l, k)
    pi_ref = np.full(k, 1 / k)
    betas = np.logspace(-1.5, 1, 30)
    kls, true_r = [], []
    for b in betas:
        pi = rlhf.kl_regularised_policy(r_hat, pi_ref, b)
        kls.append(rlhf.kl(pi, pi_ref)); true_r.append(float(pi @ r_true))
    out["kl_tradeoff"] = {"reference_policy_true_reward": round(float(pi_ref @ r_true), 3),
                          "best_possible_true_reward": round(float(r_true.max()), 3),
                          "beta=1: true_reward": round(float(rlhf.kl_regularised_policy(r_hat, pi_ref, 1.0) @ r_true), 3),
                          "beta=1: KL": round(rlhf.kl(rlhf.kl_regularised_policy(r_hat, pi_ref, 1.0), pi_ref), 3)}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].semilogx(ns, np.array(pair_acc) * 100, "o-")
    ax[0].set(xlabel="# pairwise comparisons", ylabel="% pairs ranked correctly", title="Bradley–Terry reward model")
    ax[1].plot(kls, true_r, "o-", ms=3)
    ax[1].axhline(r_true.max(), ls="--", c="k", lw=1, label="best response")
    ax[1].set(xlabel="KL(π ‖ π_ref)", ylabel="true expected reward", title="KL-regularised policy (β from 10 → 0.03)")
    ax[1].legend()
    save("rlhf")
    summary["week7_rlhf"] = out


# ------------------------------------------------------------------ Week 7: MARL / game theory
def week7_marl():
    out = {}
    A, B = marl.PRISONERS_DILEMMA
    out["prisoners_dilemma_pure_nash"] = [("C" if a == 0 else "D", "C" if b == 0 else "D") for a, b in marl.pure_nash(A, B)]
    acts = marl.independent_q(A, B)
    out["prisoners_dilemma_independent_q_defect_rate_last5000"] = round(float((acts[-5000:] == 1).mean()), 3)
    A, B = marl.MATCHING_PENNIES
    out["matching_pennies_pure_nash"] = marl.pure_nash(A, B)
    p, q = marl.mixed_nash_2x2(A, B)
    out["matching_pennies_mixed_nash_p_heads"] = [float(p), float(q)]
    freq = marl.fictitious_play(A, B)
    out["matching_pennies_fictitious_play_final_freq"] = [round(float(x), 3) for x in freq[-1]]
    plt.figure(figsize=(6.5, 4))
    plt.semilogx(np.arange(1, len(freq) + 1), freq[:, 0], label="player 1: P(heads)")
    plt.semilogx(np.arange(1, len(freq) + 1), freq[:, 1], label="player 2: P(heads)")
    plt.axhline(0.5, ls="--", c="k", lw=1, label="mixed Nash = 0.5")
    plt.xlabel("round"); plt.ylabel("empirical frequency"); plt.legend()
    plt.title("Fictitious play converges to the mixed Nash equilibrium")
    save("marl")
    summary["week7_marl"] = out


if __name__ == "__main__":
    for f in (week2_bandits, week2_bellman, week3_dp, week3_td, week7_rlhf, week7_marl):
        t = time.time(); f(); print(f"{f.__name__}: {time.time() - t:.1f}s")
    with open(os.path.join(OUT, "foundations_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
