"""DDPG (Lillicrap et al., 2016) for continuous control: deterministic actor, Q critic, soft target updates."""
import copy

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.common import mlp, set_seed


def train_ddpg(env_id="Pendulum-v1", total_steps=30_000, seed=0, gamma=0.99, tau=0.005,
               actor_lr=1e-3, critic_lr=1e-3, batch_size=128, buffer_size=100_000,
               learning_starts=1_000, noise_std=0.1):
    set_seed(seed)
    env = gym.make(env_id)
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    a_max = float(env.action_space.high[0])

    actor = mlp(obs_dim, act_dim, (256, 256), out_act=nn.Tanh)  # outputs in [-1, 1], scaled by a_max
    critic = mlp(obs_dim + act_dim, 1, (256, 256))
    actor_t, critic_t = copy.deepcopy(actor), copy.deepcopy(critic)
    a_opt = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    c_opt = torch.optim.Adam(critic.parameters(), lr=critic_lr)

    S = np.zeros((buffer_size, obs_dim), np.float32); A = np.zeros((buffer_size, act_dim), np.float32)
    R = np.zeros(buffer_size, np.float32); S2 = np.zeros_like(S); D = np.zeros(buffer_size, np.float32)
    ptr = size = 0

    episode_returns, ep_ret = [], 0.0
    s, _ = env.reset(seed=seed)
    for t in range(total_steps):
        if t < learning_starts:
            a = env.action_space.sample()
        else:
            with torch.no_grad():
                a = actor(torch.as_tensor(s, dtype=torch.float32)).numpy() * a_max
            a = np.clip(a + np.random.normal(0, noise_std * a_max, act_dim), -a_max, a_max)
        s2, r, terminated, truncated, _ = env.step(a)
        S[ptr], A[ptr], R[ptr], S2[ptr], D[ptr] = s, a / a_max, r, s2, float(terminated)
        ptr, size = (ptr + 1) % buffer_size, min(size + 1, buffer_size)
        s, ep_ret = s2, ep_ret + r
        if terminated or truncated:
            episode_returns.append(ep_ret); s, _ = env.reset(); ep_ret = 0.0

        if t >= learning_starts:
            i = np.random.randint(0, size, batch_size)
            bs, ba, br, bs2, bd = (torch.as_tensor(x[i]) for x in (S, A, R, S2, D))
            with torch.no_grad():
                y = br + gamma * (1 - bd) * critic_t(torch.cat([bs2, actor_t(bs2)], 1)).squeeze(1)
            c_loss = F.mse_loss(critic(torch.cat([bs, ba], 1)).squeeze(1), y)
            c_opt.zero_grad(); c_loss.backward(); c_opt.step()

            a_loss = -critic(torch.cat([bs, actor(bs)], 1)).mean()  # deterministic policy gradient
            a_opt.zero_grad(); a_loss.backward(); a_opt.step()

            with torch.no_grad():  # Polyak averaging of target networks
                for net, tgt in ((actor, actor_t), (critic, critic_t)):
                    for p, pt in zip(net.parameters(), tgt.parameters()):
                        pt.mul_(1 - tau).add_(tau * p)
    env.close()
    return actor, episode_returns
