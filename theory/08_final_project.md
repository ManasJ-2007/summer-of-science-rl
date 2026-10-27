# Week 8: Final project, what actually makes deep RL work?

**Question.** Which design choices matter for value-based and policy-based deep RL, and how sensitive are they to hyperparameters and random seeds?

**Setup.**
- Environments: CartPole-v1, the main testbed (discrete, 4-dim state, solved at 475+); LunarLander-v3 (8-dim state, 4 actions, solved at ≥ 200); Pendulum-v1 (continuous action).
- Every configuration is trained with **3 seeds**.
- Results report training return (last 20 episodes) and **greedy evaluation** over 20 fresh episodes, as mean ± std.

**Experiments**

| # | Experiment | Script |
|---|---|---|
| 1 | DQN component ablation: no replay / no target network / Double DQN | `run_deep.py ablation` |
| 2 | DQN hyperparameter sensitivity: ε-decay length, batch size, target-update period | `run_deep.py dqn_tuning` |
| 3 | REINFORCE with vs without a learned baseline, plus gradient-variance measurement | `run_deep.py reinforce`, `pg_variance.py` |
| 4 | Value-based vs policy-based on CartPole: DQN, REINFORCE, A2C, PPO | `run_deep.py cartpole` |
| 5 | Scaling up: REINFORCE vs A2C vs PPO on LunarLander; DDPG on Pendulum | `run_deep.py lunar lunar_pg pendulum` |

All tables and figures are in the main [README](../README.md#results).

**Observations and failure cases**
- Removing the target network does not just slow learning, it **diverges**. Q-values grew by about 5 orders of magnitude above their theoretical bound.
- The ± columns are large for DQN and A2C: **seed variance is often larger than the effect of a hyperparameter**. Reporting a single run would be misleading.
- PPO was the most robust "default" algorithm, solving CartPole and LunarLander on every seed without tuning.
- REINFORCE can reach a perfect greedy policy while its training curve still oscillates, because the policy keeps being kicked by high-variance updates.

**Limitations**
- Only 3 seeds and small compute budgets (CPU), so confidence intervals are wide.
- One-at-a-time hyperparameter sweeps miss interactions between settings.
- Same network sizes everywhere, with no tuning per algorithm beyond standard defaults.
