"""Deep Q-Network (Mnih et al., 2015) with experience replay and a target network.

The flags `use_replay` and `use_target` exist for the ablation study: turning either off
reproduces the instability that motivated these two tricks.
"""
import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from rl.common import mlp, set_seed


class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def push(self, *transition):
        self.buf.append(transition)

    def sample(self, n):
        s, a, r, s2, d = zip(*random.sample(self.buf, n))
        return (torch.as_tensor(np.array(s), dtype=torch.float32), torch.as_tensor(a),
                torch.as_tensor(r, dtype=torch.float32), torch.as_tensor(np.array(s2), dtype=torch.float32),
                torch.as_tensor(d, dtype=torch.float32))

    def __len__(self):
        return len(self.buf)


def train_dqn(env_id="CartPole-v1", total_steps=60_000, seed=0, gamma=0.99, lr=5e-4,
              batch_size=64, buffer_size=50_000, learning_starts=1_000, train_freq=1,
              target_update=500, eps_start=1.0, eps_end=0.02, eps_frac=0.2,
              use_replay=True, use_target=True, double=False):
    set_seed(seed)
    env = gym.make(env_id)
    obs_dim, n_actions = env.observation_space.shape[0], env.action_space.n
    q = mlp(obs_dim, n_actions)
    q_target = mlp(obs_dim, n_actions)
    q_target.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=lr)
    # without replay we still keep a tiny buffer: the most recent `batch_size` transitions (correlated)
    buffer = ReplayBuffer(buffer_size if use_replay else batch_size)

    episode_returns, ep_ret = [], 0.0
    s, _ = env.reset(seed=seed)
    for t in range(total_steps):
        eps = max(eps_end, eps_start - (eps_start - eps_end) * t / (eps_frac * total_steps))
        if random.random() < eps:
            a = env.action_space.sample()
        else:
            with torch.no_grad():
                a = int(q(torch.as_tensor(s, dtype=torch.float32)).argmax())
        s2, r, terminated, truncated, _ = env.step(a)
        # store `terminated` (not truncated): a time-limit cut is not a true terminal state
        buffer.push(s, a, r, s2, float(terminated))
        s, ep_ret = s2, ep_ret + r
        if terminated or truncated:
            episode_returns.append(ep_ret)
            s, _ = env.reset()
            ep_ret = 0.0

        if t >= learning_starts and t % train_freq == 0 and len(buffer) >= batch_size:
            bs, ba, br, bs2, bd = buffer.sample(batch_size)
            with torch.no_grad():
                net = q_target if use_target else q
                if double:  # Double DQN: online net picks the action, target net evaluates it
                    a2 = q(bs2).argmax(1, keepdim=True)
                    next_q = net(bs2).gather(1, a2).squeeze(1)
                else:
                    next_q = net(bs2).max(1).values
                target = br + gamma * (1 - bd) * next_q
            pred = q(bs).gather(1, ba.unsqueeze(1)).squeeze(1)
            loss = F.smooth_l1_loss(pred, target)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q.parameters(), 10.0)
            opt.step()

        if use_target and t % target_update == 0:
            q_target.load_state_dict(q.state_dict())

    env.close()
    return q, episode_returns
