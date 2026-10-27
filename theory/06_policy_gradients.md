# Week 6: Policy-gradient methods (REINFORCE, A2C, PPO)

Instead of learning values and acting greedily, parameterise the policy $\pi_\theta(a \mid s)$ directly and climb $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[G_0]$. This works for stochastic and continuous policies, and small parameter changes give smooth changes in behaviour.

## Policy-gradient theorem (derivation)
$$\nabla_\theta J = \nabla_\theta \sum_\tau p_\theta(\tau) G(\tau) = \sum_\tau p_\theta(\tau)\,\nabla_\theta \log p_\theta(\tau)\, G(\tau) = \mathbb{E}_\tau\Big[\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)\, G(\tau)\Big].$$

The second step is the log-derivative trick. The dynamics terms of $\log p_\theta(\tau)$ do not depend on θ (Week 1). Rewards earned before time *t* cannot depend on $a_t$ (causality), so $G(\tau)$ can be replaced by the reward-to-go $G_t$. That gives **REINFORCE**:

$$\nabla_\theta J \approx \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)\, G_t .$$

## Baselines: unbiased variance reduction
For any $b(s)$ that does not depend on the action,
$\mathbb{E}_{a \sim \pi}[\nabla_\theta \log \pi_\theta(a \mid s)\, b(s)] = b(s)\,\nabla_\theta \sum_a \pi_\theta(a \mid s) = b(s)\,\nabla_\theta 1 = 0$.
So $G_t - b(s_t)$ gives the same expected gradient. With $b = V(s)$, the weight becomes the **advantage** $A_t = G_t - V(s_t)$: "how much better than usual was this action?"

**Measured** (`scripts/pg_variance.py`, CartPole, 500 sampled episodes per policy, gradient of the policy network):

| Policy being evaluated | tr Cov[g], no baseline | tr Cov[g], with V(s) baseline | Variance reduction | Cosine sim. of mean gradients |
|---|---|---|---|---|
| Early (100 episodes, return ≈ 114) | 2.63 × 10⁶ | 1.89 × 10⁶ | **28.0%** | 1.000 |
| Near-solved (300 episodes, return ≈ 456) | 2.04 × 10⁷ | 9.48 × 10⁶ | **53.7%** | 0.993 |

The baseline leaves the expected gradient unchanged (cosine similarity ≈ 1, as the proof above predicts) but removes up to **54% of its variance**. The effect grows as returns get longer, because $G_t$ then has a large common offset that the baseline subtracts out.

## Actor-critic (A2C)
Replace the Monte-Carlo return with a bootstrapped estimate, using the critic's TD error as the advantage: $A_t \approx r_t + \gamma V(s_{t+1}) - V(s_t)$. It updates every *n* steps, with no need to wait for the episode to end. It has lower variance than REINFORCE but is biased by the critic. **GAE(λ)** interpolates between the two: $\hat A_t = \sum_l (\gamma\lambda)^l \delta_{t+l}$.

## PPO: clipped surrogate objective
A large policy step can collapse performance. PPO maximises

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t\Big[\min\big(\rho_t \hat A_t,\ \mathrm{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\,\hat A_t\big)\Big], \qquad \rho_t = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_\text{old}}(a_t \mid s_t)} .$$

Once $\rho_t$ leaves $[1-\epsilon, 1+\epsilon]$ in the direction that would increase the objective, the gradient is zero. That gives a cheap trust region, so several epochs of mini-batch SGD on the same rollout are safe. The implementation in `rl/policy_gradient.py` uses ε = 0.2, GAE λ = 0.95, 10 epochs, advantage normalisation and learning-rate annealing.

## Value-based vs policy-based: results
See the main README for CartPole (DQN vs REINFORCE vs A2C vs PPO) and LunarLander (REINFORCE vs A2C vs PPO). In brief:
- **PPO** solved CartPole on every seed with a stable curve, in a median of 38.7k steps, and was the **only** method to solve LunarLander on every seed (greedy 245 ± 26).
- **REINFORCE** reached a perfect greedy CartPole policy but oscillated during training. On LunarLander it reached 108 ± 61.
- **Single-environment A2C** failed on LunarLander at this budget (9 ± 24).
- **DQN** was the most seed-sensitive.

*Reading: Sutton & Barto, Ch. 13; Schulman et al., "Proximal Policy Optimization Algorithms" (2017) and "High-Dimensional Continuous Control Using GAE" (2016).*
