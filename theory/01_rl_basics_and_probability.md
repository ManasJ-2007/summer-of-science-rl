# Week 1: RL orientation and math refresher

## The RL problem
An **agent** interacts with an **environment** in discrete time. At step *t* it observes a **state** $S_t$, picks an **action** $A_t$, and receives a **reward** $R_{t+1}$ and the next state $S_{t+1}$. A **trajectory** is $\tau = (S_0, A_0, R_1, S_1, A_1, R_2, \dots)$. An **episode** is a trajectory that ends in a terminal state.

- **Policy** $\pi(a \mid s)$: the agent's behaviour, a distribution over actions given the state.
- **Return** (discounted): $G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$, with $0 \le \gamma \le 1$. It satisfies the recursion $G_t = R_{t+1} + \gamma G_{t+1}$, which every Bellman equation builds on.
- **Value functions:** $v_\pi(s) = \mathbb{E}_\pi[G_t \mid S_t = s]$ and $q_\pi(s,a) = \mathbb{E}_\pi[G_t \mid S_t = s, A_t = a]$.
- **Goal:** find $\pi^*$ maximising $v_\pi(s)$ for every state.

How RL differs from supervised learning: there are no labels, only a scalar reward. Feedback is **delayed** (credit assignment), and the agent's own actions change the data it sees (**exploration vs exploitation**).

## Probability refresher
| Concept | Formula | Where it appears in RL |
|---|---|---|
| Expectation | $\mathbb{E}[X] = \sum_x x\,p(x)$ | value functions are expected returns |
| Linearity | $\mathbb{E}[aX + bY] = a\mathbb{E}[X] + b\mathbb{E}[Y]$ | splitting $G_t = R_{t+1} + \gamma G_{t+1}$ inside $\mathbb{E}$ |
| Law of total expectation | $\mathbb{E}[X] = \mathbb{E}_Y[\mathbb{E}[X \mid Y]]$ | deriving the Bellman equation (condition on $A_t$, then $S_{t+1}$) |
| Chain rule | $p(x, y) = p(x)\,p(y \mid x)$ | trajectory probability $p(\tau) = p(s_0)\prod_t \pi(a_t \mid s_t)\,p(s_{t+1} \mid s_t, a_t)$ |
| Bayes' theorem | $p(y \mid x) = \dfrac{p(x \mid y)\,p(y)}{p(x)}$ | belief updates, Thompson sampling, Bayesian RL |
| Log-derivative trick | $\nabla_\theta p_\theta(x) = p_\theta(x)\,\nabla_\theta \log p_\theta(x)$ | the policy-gradient theorem (Week 6) |
| Variance | $\mathrm{Var}[X] = \mathbb{E}[X^2] - \mathbb{E}[X]^2$ | why Monte-Carlo returns are noisy, why baselines help |

**Worked example (chain rule).** Under the chain rule, the log-probability of a trajectory separates into a policy part and a dynamics part:

$$\log p_\theta(\tau) = \log p(s_0) + \sum_t \log \pi_\theta(a_t \mid s_t) + \sum_t \log p(s_{t+1} \mid s_t, a_t).$$

Only the policy term depends on $\theta$, so $\nabla_\theta \log p_\theta(\tau) = \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)$. This is why REINFORCE needs no model of the environment.

## Setup
`requirements.txt` installs Gymnasium (with Box2D), PyTorch, NumPy and Matplotlib. Every script in `scripts/` runs on a CPU.

*Reading: Sutton & Barto, Ch. 1; Lilian Weng, "A (Long) Peek into Reinforcement Learning".*
