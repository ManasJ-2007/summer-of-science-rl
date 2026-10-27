"""Week 7: the two core ideas of RLHF on a toy problem (k candidate "responses" = arms).

1. Reward modelling from pairwise preferences (Bradley-Terry):
       P(i preferred over j) = sigmoid(r_i - r_j)
   fit r_hat by maximising the log-likelihood of observed comparisons (logistic regression).
2. KL-regularised policy optimisation against a reference policy pi_ref:
       max_pi  E_pi[r_hat] - beta * KL(pi || pi_ref)
   whose closed-form solution is  pi*(a) ∝ pi_ref(a) * exp(r_hat(a) / beta).
"""
import numpy as np


def simulate_preferences(r_true, n_pairs, rng):
    k = len(r_true)
    i = rng.integers(0, k, n_pairs)
    j = (i + rng.integers(1, k, n_pairs)) % k  # j != i
    p = 1 / (1 + np.exp(-(r_true[i] - r_true[j])))
    i_wins = rng.random(n_pairs) < p
    return np.where(i_wins, i, j), np.where(i_wins, j, i)  # (winner, loser)


def fit_bradley_terry(winners, losers, k, lr=0.5, iters=2000, l2=1e-3):
    """Gradient ascent on sum log sigmoid(r_w - r_l). Rewards are identifiable only up to a constant."""
    r = np.zeros(k)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(r[winners] - r[losers])))
        g = np.zeros(k)
        np.add.at(g, winners, 1 - p)
        np.add.at(g, losers, -(1 - p))
        r += lr * (g / len(winners) - l2 * r)
    return r - r.mean()


def kl_regularised_policy(r_hat, pi_ref, beta):
    logits = np.log(pi_ref) + r_hat / beta
    pi = np.exp(logits - logits.max())
    return pi / pi.sum()


def kl(p, q):
    return float(np.sum(p * (np.log(p + 1e-300) - np.log(q))))
