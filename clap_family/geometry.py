"""Discrete latent-trajectory geometry (paper Eq. 4-7, 22)."""
import numpy as np

from .constants import EPS


def velocity(z, dt):
    """Forward finite-difference velocity v_t = (z_{t+1} - z_t)/dt. Shape (H, d)."""
    z = np.asarray(z, dtype=float)
    return (z[1:] - z[:-1]) / dt


def acceleration(z, dt):
    """Central finite-difference acceleration a_t = (z_{t+1} - 2 z_t + z_{t-1})/dt^2. Shape (H-1, d)."""
    z = np.asarray(z, dtype=float)
    return (z[2:] - 2.0 * z[1:-1] + z[:-2]) / (dt ** 2)


def norm_sq(v, G=None):
    """Squared Riemannian norm ||v||^2_G (Eq. 5). G=None means identity; G may be a (d,d) matrix."""
    v = np.asarray(v, dtype=float)
    if G is None:
        return np.sum(v * v, axis=-1)
    Gv = v @ np.asarray(G, dtype=float)
    return np.sum(v * Gv, axis=-1)


def speed_ratio_q(v, c, G=None, eps=EPS):
    """Normalized speed q = ||v||^2_G / c(z)^2 (Eq. 4). v:(H,d), c:(H,)."""
    return norm_sq(v, G) / (np.asarray(c, dtype=float) ** 2 + eps)
