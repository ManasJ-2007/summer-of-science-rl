"""Week 3: TD(0) vs constant-alpha Monte-Carlo prediction on the 5-state random walk (Sutton & Barto, Example 6.2).

States A..E sit between two terminals; start in C, step left/right with probability 1/2.
Reward +1 only on exiting right, gamma = 1, so the true values are v = [1, 2, 3, 4, 5] / 6.
"""
import numpy as np

TRUE_V = np.arange(1, 6) / 6.0


def episode(rng):
    s, states = 2, [2]
    while True:
        s += 1 if rng.random() < 0.5 else -1
        if s < 0:
            return states, 0.0
        if s > 4:
            return states, 1.0
        states.append(s)


def rms_curve(method, alpha, episodes=100, runs=100, seed=0):
    """RMS error (averaged over states and runs) after each episode, starting from V = 0.5."""
    rng = np.random.default_rng(seed)
    err = np.zeros(episodes)
    for _ in range(runs):
        V = np.full(5, 0.5)
        for ep in range(episodes):
            states, final_r = episode(rng)
            if method == "td":
                for i, s in enumerate(states):
                    if i + 1 < len(states):
                        target = V[states[i + 1]]   # r = 0 inside the chain
                    else:
                        target = final_r            # transition into a terminal state
                    V[s] += alpha * (target - V[s])
            else:  # every-visit MC: the return from every state is the final reward
                for s in states:
                    V[s] += alpha * (final_r - V[s])
            err[ep] += np.sqrt(np.mean((V - TRUE_V) ** 2))
    return err / runs
