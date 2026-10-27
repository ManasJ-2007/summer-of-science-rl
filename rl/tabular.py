"""Tabular RL: value iteration (model-based) and Q-learning / SARSA (model-free TD control)."""
import numpy as np


def epsilon_greedy(Q, s, eps, rng):
    if rng.random() < eps:
        return int(rng.integers(Q.shape[1]))
    # break ties randomly so an all-zero Q-table doesn't always pick action 0
    best = np.flatnonzero(Q[s] == Q[s].max())
    return int(rng.choice(best))


def linear_decay(start, end, frac, total):
    """Returns eps(t): linear from start to end over the first `frac` of `total` episodes."""
    horizon = max(1, int(frac * total))
    return lambda t: end + (start - end) * max(0.0, 1 - t / horizon)


def td_control(env, method, episodes, alpha=0.1, gamma=0.99,
               eps_start=1.0, eps_end=0.01, eps_frac=0.6, seed=0):
    """Q-learning (off-policy) or SARSA (on-policy).

    Q-learning target:  r + gamma * max_a' Q(s', a')
    SARSA target:       r + gamma * Q(s', a')   where a' is the action actually taken next
    """
    assert method in ("qlearning", "sarsa")
    rng = np.random.default_rng(seed)
    nS, nA = env.observation_space.n, env.action_space.n
    Q = np.zeros((nS, nA))
    eps_at = linear_decay(eps_start, eps_end, eps_frac, episodes)
    returns = np.zeros(episodes)

    for ep in range(episodes):
        eps = eps_at(ep)
        s, _ = env.reset(seed=seed + ep)
        a = epsilon_greedy(Q, s, eps, rng)
        done, G = False, 0.0
        while not done:
            s2, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            a2 = epsilon_greedy(Q, s2, eps, rng)
            if terminated:
                target = r  # no bootstrapping from a terminal state
            elif method == "qlearning":
                target = r + gamma * Q[s2].max()
            else:
                target = r + gamma * Q[s2, a2]
            Q[s, a] += alpha * (target - Q[s, a])
            s, a, G = s2, a2, G + r
        returns[ep] = G
    return Q, returns


def value_iteration(env, gamma=0.99, tol=1e-8):
    """Bellman optimality backups using the known transition model env.unwrapped.P."""
    P = env.unwrapped.P
    nS, nA = env.observation_space.n, env.action_space.n
    V = np.zeros(nS)
    sweeps = 0
    while True:
        sweeps += 1
        Q = np.zeros((nS, nA))
        for s in range(nS):
            for a in range(nA):
                Q[s, a] = sum(p * (r + gamma * V[s2] * (not term)) for p, s2, r, term in P[s][a])
        V_new = Q.max(axis=1)
        if np.max(np.abs(V_new - V)) < tol:
            return Q, V_new, sweeps
        V = V_new


def evaluate_greedy(env, Q, episodes=1000, seed=10_000):
    """Mean return of the greedy policy argmax_a Q(s, a)."""
    total = 0.0
    for ep in range(episodes):
        s, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            s, r, terminated, truncated, _ = env.step(int(np.argmax(Q[s])))
            done = terminated or truncated
            total += r
    return total / episodes
