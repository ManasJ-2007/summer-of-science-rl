# Week 3: Dynamic programming and temporal-difference learning

## Policy iteration (`rl/dp.py`)
1. **Policy evaluation:** compute $v_\pi$ exactly with $(I - \gamma P_\pi)^{-1} r_\pi$.
2. **Policy improvement:** $\pi'(s) = \arg\max_a \big[r(s,a) + \gamma \sum_{s'} p(s' \mid s,a)\,v_\pi(s')\big]$.
3. Repeat until the policy stops changing.

**Policy improvement theorem.** If $q_\pi(s, \pi'(s)) \ge v_\pi(s)$ for all $s$, then $v_{\pi'} \ge v_\pi$. The greedy policy satisfies this by construction. A finite MDP has finitely many deterministic policies, so policy iteration terminates at $\pi^*$.

## Value iteration (`rl/tabular.py`)
Apply the Bellman optimality operator directly: $v_{k+1}(s) = \max_a \sum_{s'} p(s' \mid s,a)\,[r + \gamma v_k(s')]$. Each sweep is cheap, but convergence is only geometric, at rate γ.

| FrozenLake (slippery, γ = 0.99) | Policy iteration | Value iteration (tol 1e-10) | Max \|v_PI − v_VI\| | Optimal success rate |
|---|---|---|---|---|
| 4×4 | **3 improvement steps** | 571 sweeps | 2.8 × 10⁻⁹ | 72.9% |
| 8×8 | **3 improvement steps** | 662 sweeps | 3.1 × 10⁻⁹ | 62.7% |

Both methods reach the same $v_*$, to numerical precision. Policy iteration takes very few (expensive) steps, while value iteration takes many cheap ones. Even the *optimal* policy reaches the goal only 73% of the time on the 4×4 map, because the ice is slippery: the intended move happens with probability 1/3 only.

![frozenlake](../results/foundations_frozenlake_policy.png)

## Temporal-difference learning (`rl/td.py`)
Without a model, estimate $v_\pi$ from experience:

- **Monte Carlo:** $V(S_t) \leftarrow V(S_t) + \alpha\,[G_t - V(S_t)]$. The target is the full return, which is unbiased but high-variance and has to wait for the episode to end.
- **TD(0):** $V(S_t) \leftarrow V(S_t) + \alpha\,[R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]$. It **bootstraps** from its own estimate: lower variance, biased early on, and it can update online.

The bracket $\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$ is the **TD error**. SARSA, Q-learning, DQN and actor-critic all minimise versions of it.

**Random-walk experiment** (Sutton & Barto, Example 6.2; true values 1/6 … 5/6; 100 runs):

| Method | Best α | RMS error after 100 episodes |
|---|---|---|
| **TD(0)** | 0.05 | **0.038** |
| Constant-α MC | 0.02 | 0.077 |

TD(0) reaches **51% lower error** than the best Monte-Carlo setting with the same data. Bootstrapping exploits the Markov structure (C's value really is the average of B's and D's), while MC treats every state's return independently.

![td vs mc](../results/foundations_td_vs_mc.png)

*Reading: Sutton & Barto, Ch. 4–6.*
