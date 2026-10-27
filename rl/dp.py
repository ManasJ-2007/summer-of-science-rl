"""Weeks 2-3: Bellman equations and dynamic programming on a known MDP (Gymnasium's env.unwrapped.P).

- policy_evaluation_exact:     solve  v = r_pi + gamma * P_pi v   as a linear system
- policy_evaluation_iterative: repeated Bellman expectation backups (a gamma-contraction)
- policy_iteration:            evaluate -> act greedily, until the policy stops changing
- value_iteration lives in rl/tabular.py
"""
import numpy as np


def mdp_matrices(env):
    """Transition tensor P[s, a, s'] and expected reward R[s, a] from Gymnasium's model."""
    model = env.unwrapped.P
    nS, nA = env.observation_space.n, env.action_space.n
    P, R = np.zeros((nS, nA, nS)), np.zeros((nS, nA))
    for s in range(nS):
        for a in range(nA):
            for p, s2, r, term in model[s][a]:
                R[s, a] += p * r
                if not term:  # a terminal next state contributes no future value
                    P[s, a, s2] += p
    return P, R


def policy_matrices(P, R, pi):
    """pi[s, a] = probability of action a in state s  ->  (P_pi[s, s'], r_pi[s])."""
    return np.einsum("sa,sat->st", pi, P), (pi * R).sum(1)


def policy_evaluation_exact(P, R, pi, gamma):
    P_pi, r_pi = policy_matrices(P, R, pi)
    return np.linalg.solve(np.eye(len(r_pi)) - gamma * P_pi, r_pi)


def policy_evaluation_iterative(P, R, pi, gamma, tol=1e-10, max_iter=100_000):
    """Returns (v, list of sup-norm changes per sweep). The changes shrink by at most gamma per sweep."""
    P_pi, r_pi = policy_matrices(P, R, pi)
    v, deltas = np.zeros(len(r_pi)), []
    for _ in range(max_iter):
        v_new = r_pi + gamma * P_pi @ v
        deltas.append(np.max(np.abs(v_new - v)))
        v = v_new
        if deltas[-1] < tol:
            break
    return v, deltas


def greedy_policy(P, R, v, gamma):
    nS, nA = R.shape
    q = R + gamma * P @ v
    pi = np.zeros((nS, nA))
    pi[np.arange(nS), q.argmax(1)] = 1.0
    return pi, q


def policy_iteration(P, R, gamma):
    """Returns (optimal deterministic policy, v*, number of improvement steps)."""
    nS, nA = R.shape
    pi = np.full((nS, nA), 1.0 / nA)  # start from the uniform random policy
    for it in range(1, 1000):
        v = policy_evaluation_exact(P, R, pi, gamma)
        new_pi, _ = greedy_policy(P, R, v, gamma)
        if np.array_equal(new_pi.argmax(1), pi.argmax(1)) and it > 1:
            return new_pi, v, it
        pi = new_pi
    return pi, v, it
