"""Environment protocol and a SimpleEnv container.

An Env supplies the fields, a start state, the target manifold, N_star, and the
single-integrator dynamics z_{t+1} = z_t + dt * v used by the planner. Region predicates
(in_target/in_trap/in_unsafe) drive the theorem-aligned metrics.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..admissible import dist_to_target


@dataclass
class SimpleEnv:
    fields: object
    start: np.ndarray
    target: np.ndarray          # (k, d) target states M*
    N_star: float = 1.0
    target_radius: float = 0.3
    trap_center: Optional[np.ndarray] = None
    trap_radius: float = 0.3
    unsafe_center: Optional[np.ndarray] = None
    unsafe_radius: float = 0.3
    shortcut_center: Optional[np.ndarray] = None

    def __post_init__(self):
        self.start = np.asarray(self.start, dtype=float)
        self.target = np.atleast_2d(np.asarray(self.target, dtype=float))

    def in_target(self, z):
        return dist_to_target(z, self.target) <= self.target_radius

    def _in_region(self, z, center, radius):
        if center is None:
            z = np.atleast_2d(np.asarray(z, dtype=float))
            return np.zeros(z.shape[0], dtype=bool)
        z = np.atleast_2d(np.asarray(z, dtype=float))
        return np.linalg.norm(z - np.asarray(center, dtype=float), axis=-1) <= radius

    def in_trap(self, z):
        return self._in_region(z, self.trap_center, self.trap_radius)

    def in_unsafe(self, z):
        return self._in_region(z, self.unsafe_center, self.unsafe_radius)
