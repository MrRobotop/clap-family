"""Gymnasium + Stable-Baselines3 adapter. Requires `pip install clap_family[rl]`.

CLAPRewardWrapper shapes the environment reward by the conservative lapse N̄0 at the next state,
so any SB3 algorithm trains toward safe high-lapse regions instead of raw reward.
"""
import gymnasium as gym
import numpy as np

from ..lapse import conservative_lapse
from ..params import CLAPParams


class CLAPRewardWrapper(gym.Wrapper):
    """Wrap a Gymnasium env to augment reward with the CLAP conservative lapse at the next state.

    shaped_reward = raw_reward + lapse_weight * N̄0(next_obs).
    """

    def __init__(self, env, fields, params=None, lapse_weight=1.0, **kwargs):
        super().__init__(env)
        self.fields = fields
        self.params = params if params is not None else CLAPParams(**kwargs)
        self.lapse_weight = lapse_weight

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        p = self.params
        z = np.atleast_2d(np.asarray(obs, dtype=float))
        n_bar = conservative_lapse(self.fields.V(z), self.fields.U(z), self.fields.C(z),
                                   self.fields.O(z), p.beta, p.eta, p.zeta, p.tau)[0]
        shaped = reward + self.lapse_weight * float(n_bar)
        info = dict(info, clap_lapse=float(n_bar), raw_reward=float(reward))
        return obs, shaped, terminated, truncated, info
