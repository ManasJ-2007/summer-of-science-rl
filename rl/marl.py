"""Week 7: Game theory 101 and multi-agent learning in 2-player matrix games.

- pure_nash:            brute-force pure-strategy Nash equilibria of a bimatrix game
- mixed_nash_2x2:       closed-form mixed equilibrium of a 2x2 game (indifference conditions)
- independent_q:        two independent epsilon-greedy Q-learners (stateless repeated game)
- fictitious_play:      each player best-responds to the opponent's empirical action frequencies
"""
import numpy as np

# payoffs[player][my_action, opponent_action]; actions: 0 = cooperate, 1 = defect
PRISONERS_DILEMMA = (np.array([[3, 0], [5, 1]]), np.array([[3, 0], [5, 1]]))
# matching pennies: player 1 wins on a match, player 2 on a mismatch (zero-sum)
MATCHING_PENNIES = (np.array([[1, -1], [-1, 1]]), np.array([[-1, 1], [1, -1]]))


def pure_nash(A, B):
    """A[a1, a2] = payoff of player 1, B[a2, a1] = payoff of player 2 (each indexed by own action first)."""
    eq = []
    for a1 in range(A.shape[0]):
        for a2 in range(A.shape[1]):
            if A[a1, a2] >= A[:, a2].max() and B[a2, a1] >= B[:, a1].max():
                eq.append((a1, a2))
    return eq


def mixed_nash_2x2(A, B):
    """Probability each player puts on action 0 at the fully mixed equilibrium.

    Player 2 mixes with q so that player 1 is indifferent:  q*A00 + (1-q)*A01 = q*A10 + (1-q)*A11.
    """
    q = (A[1, 1] - A[0, 1]) / (A[0, 0] - A[0, 1] - A[1, 0] + A[1, 1])
    p = (B[1, 1] - B[0, 1]) / (B[0, 0] - B[0, 1] - B[1, 0] + B[1, 1])
    return p, q


def independent_q(A, B, steps=20_000, alpha=0.05, eps=0.1, seed=0):
    """Returns the per-step probability (over a sliding window) that each agent plays action 0."""
    rng = np.random.default_rng(seed)
    Q1, Q2 = np.zeros(2), np.zeros(2)
    acts = np.zeros((steps, 2), dtype=int)
    for t in range(steps):
        a1 = rng.integers(2) if rng.random() < eps else int(Q1.argmax())
        a2 = rng.integers(2) if rng.random() < eps else int(Q2.argmax())
        Q1[a1] += alpha * (A[a1, a2] - Q1[a1])
        Q2[a2] += alpha * (B[a2, a1] - Q2[a2])
        acts[t] = a1, a2
    return acts


def fictitious_play(A, B, steps=20_000):
    """Returns the running empirical frequency of action 0 for both players."""
    c1, c2 = np.ones(2), np.ones(2)  # action counts (start at 1 to avoid division by zero)
    freq = np.zeros((steps, 2))
    for t in range(steps):
        a1 = int((A @ (c2 / c2.sum())).argmax())   # best response to player 2's empirical mix
        a2 = int((B @ (c1 / c1.sum())).argmax())
        c1[a1] += 1; c2[a2] += 1
        freq[t] = c1[0] / c1.sum(), c2[0] / c2.sum()
    return freq
