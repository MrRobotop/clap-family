"""Conservative lapse field and target-value estimation (paper Eq. 1, 2)."""
import numpy as np


def sigmoid(x):
    """Numerically stable logistic squash sigma()."""
    x = np.clip(np.asarray(x, dtype=float), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def conservative_lapse(V, U, C, O, beta, eta, zeta, tau):  # noqa: E741
    """N̄0 = sigma((V - beta U - eta C - zeta O)/tau)  (Eq. 1)."""
    if tau <= 0:
        raise ValueError("tau must be > 0")
    V, U, C, O = (np.asarray(x, dtype=float) for x in (V, U, C, O))  # noqa: E741
    return sigmoid((V - beta * U - eta * C - zeta * O) / tau)


def estimate_N_star(fields, admissible, grid, params, atol=1e-9):
    """Estimate N̄* = max_{z in X} N̄0(z) over a candidate grid and return target states M*.

    Returns (N_star: float, target_states: ndarray (k, d)). Raises ValueError if no
    grid point is admissible (Eq. 2, requires admissible set nonempty).
    """
    grid = np.atleast_2d(np.asarray(grid, dtype=float))
    mask = admissible.contains(grid, fields)
    if not np.any(mask):
        raise ValueError("admissible set is empty over the provided grid")
    z = grid[mask]
    nb = conservative_lapse(fields.V(z), fields.U(z), fields.C(z), fields.O(z),
                            params.beta, params.eta, params.zeta, params.tau)
    n_star = float(np.max(nb))
    targets = z[nb >= n_star - atol]
    return n_star, targets
