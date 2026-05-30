"""Admissible state class X and target-manifold distance (paper Eq. 8, terminal term Eq. 3)."""
from dataclasses import dataclass

import numpy as np


@dataclass
class AdmissibleSet:
    """X = { z : C(z) <= Cmax, U(z) <= Umax, O(z) <= Omax }  (Eq. 8). Defaults: no constraint."""
    Cmax: float = np.inf
    Umax: float = np.inf
    Omax: float = np.inf

    def contains(self, z, fields):
        """Boolean mask (N,) of admissibility for states z:(N,d)."""
        z = np.atleast_2d(np.asarray(z, dtype=float))
        return ((fields.C(z) <= self.Cmax)
                & (fields.U(z) <= self.Umax)
                & (fields.O(z) <= self.Omax))


def dist_to_target(z, target_states):
    """Euclidean distance d_G(z, M*) to the nearest target state. z:(N,d) or (d,)."""
    z = np.atleast_2d(np.asarray(z, dtype=float))
    t = np.atleast_2d(np.asarray(target_states, dtype=float))
    if t.shape[0] == 0:
        raise ValueError("target_states is empty")
    d = np.linalg.norm(z[:, None, :] - t[None, :, :], axis=-1)
    return np.min(d, axis=-1)
