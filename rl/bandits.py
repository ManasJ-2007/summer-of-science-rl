"""Week 2: k-armed bandit testbed (Sutton & Barto, Section 2.3) with epsilon-greedy and UCB action selection.

All runs are vectorised over independent bandit problems, so 2,000 runs x 1,000 steps takes about a second.
"""
import numpy as np


def run_testbed(eps=0.1, k=10, steps=1000, runs=2000, ucb_c=None, q0=0.0, seed=0):
    """Returns (mean reward per step, fraction of runs choosing the optimal arm per step).

    Each run draws true action values q*(a) ~ N(0, 1); a reward is q*(a) + N(0, 1).
    Estimates use the incremental sample average  Q_{n+1} = Q_n + (R_n - Q_n) / n.
    ucb_c: if set, use UCB  A_t = argmax_a [Q(a) + c * sqrt(ln t / N(a))]  instead of epsilon-greedy.
    q0: optimistic initial value (Section 2.6).
    """
    rng = np.random.default_rng(seed)
    q_star = rng.normal(0, 1, (runs, k))
    best = q_star.argmax(1)
    Q = np.full((runs, k), q0, dtype=float)
    N = np.zeros((runs, k))
    rewards, optimal = np.zeros(steps), np.zeros(steps)
    idx = np.arange(runs)
    for t in range(steps):
        if ucb_c is not None:
            bonus = ucb_c * np.sqrt(np.log(t + 1) / np.maximum(N, 1e-12))
            bonus[N == 0] = np.inf  # try every arm once first
            a = (Q + bonus).argmax(1)
        else:
            greedy = (Q + rng.random((runs, k)) * 1e-9).argmax(1)  # random tie-breaking
            explore = rng.random(runs) < eps
            a = np.where(explore, rng.integers(0, k, runs), greedy)
        r = q_star[idx, a] + rng.normal(0, 1, runs)
        N[idx, a] += 1
        Q[idx, a] += (r - Q[idx, a]) / N[idx, a]
        rewards[t] = r.mean()
        optimal[t] = (a == best).mean()
    return rewards, optimal
