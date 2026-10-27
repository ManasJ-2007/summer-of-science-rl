# Week 5: Deep Q-Networks

This note follows my DQN presentation outline: what RL is → MDPs → Q-learning → why DQN → architecture → experience replay → target network → training algorithm → applications → limitations.

## From Q-learning to DQN
Replace the table with a neural network $Q(s, a; \theta)$ and minimise the squared TD error:

$$L(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}}\Big[\big(\underbrace{r + \gamma (1 - d) \max_{a'} Q(s', a'; \theta^-)}_{y} - Q(s, a; \theta)\big)^2\Big].$$

- **Architecture (`rl/dqn.py`):** input is the 4-dim CartPole state, then two hidden layers of 128 units with ReLU, and the output is one Q-value per action. All action values come from a single forward pass.
- **Huber loss** instead of squared error, plus gradient-norm clipping at 10, so outlier TD errors cannot blow up an update.
- $d = 1$ only for true termination. A time-limit truncation is **not** terminal, so the target still bootstraps.

## Why the two tricks are needed: the "deadly triad"
Function approximation + bootstrapping + off-policy learning can diverge (Sutton & Barto, Section 11.3). DQN adds two stabilisers:

1. **Experience replay** stores transitions in a buffer and samples random mini-batches. This breaks the strong correlation between consecutive transitions, so gradient steps are closer to i.i.d., and it reuses each transition many times.
2. **Target network** $\theta^-$, a copy of $\theta$ refreshed every *C* steps. Without it, every update moves the target it is chasing.

**Double DQN** (van Hasselt et al.): the max over noisy estimates is biased upward, $\mathbb{E}[\max_a \hat Q] \ge \max_a \mathbb{E}[\hat Q]$. The fix is to let the online network *choose* the action and the target network *evaluate* it: $y = r + \gamma\,Q(s', \arg\max_{a'} Q(s',a';\theta); \theta^-)$.

## Training algorithm
```
initialise Q(θ), target θ⁻ ← θ, replay buffer D
for t = 1..T:
    a ← ε-greedy(Q(s;θ)), ε decays linearly 1.0 → 0.02 over the first 20% of steps
    step env, store (s, a, r, s', terminated) in D
    sample mini-batch from D, compute y with θ⁻, take a gradient step on Huber(y − Q(s,a;θ))
    every C steps: θ⁻ ← θ
```

## Results on CartPole-v1 (60k steps, 3 seeds; `scripts/run_deep.py ablation`)

| Variant | Training return | Greedy eval (max 500) |
|---|---|---|
| DQN (replay + target) | 292.6 ± 101.2 | 254.7 ± 178.4 |
| **Double DQN** | 265.2 ± 42.1 | **447.2 ± 40.4** |
| No replay buffer | 165.3 ± 69.4 | 139.4 ± 24.0 |
| **No target network** | **9.3 ± 0.1** | **9.2 ± 0.1** |

**The no-target-network run diverged.** At 20k steps its Q-value for the start state was about **8.4 × 10⁶**, while a correctly bounded value is at most $\sum_k \gamma^k = 100$ (the network with a target estimated about 34). The resulting policy fails in about 9 steps. That is the deadly triad in action.

Hyperparameter sensitivity (ε-decay length, batch size, target-update period *C*) is in the main README under *Week 5 tuning*.

## Applications and limitations
**Applications:** Atari from pixels (the original DQN paper), recommendation, resource scheduling, and any problem with discrete actions.
**Limitations:**
- Discrete actions only. A max over continuous actions is itself an optimisation problem, which leads to DDPG.
- Over-estimation bias, hence Double DQN.
- Sample-hungry, and sensitive to hyperparameters: results vary a lot across seeds, as the ± values above show.

*Reading: Mnih et al., "Human-level control through deep reinforcement learning" (Nature, 2015).*
