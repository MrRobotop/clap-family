"""Speed-limit projection for CEM-MPC (paper Listing 1 `project_velocity`)."""
import numpy as np

EPS = 1e-8


def project_velocity(v, cmax):
    """Scale v so ||v|| <= cmax; leave it unchanged if already within the limit."""
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)
    if norm <= cmax:
        return v
    # EPS guards the degenerate cmax->0 / tiny-norm case; for valid cmax>0 it is negligible
    return v * (cmax / (norm + EPS))
