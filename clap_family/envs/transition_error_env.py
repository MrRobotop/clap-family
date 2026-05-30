"""TransitionErrorEnv (paper §6.3): a shortcut that looks fine to static fields but has high
transition-model error E_theta. Tests learned-gate avoidance of model-unreliable regions.

Geometry
--------
start  = [-3, 0],  target = [3, 0]  — both on the x-axis.
The direct straight-line path at y ≈ 0 passes through the center of a high-E_theta
anisotropic corridor (x_width=1.5, y_width=0.3, peak amplitude=1.5 at origin).
Static fields (V, U, C, O) look completely benign along the direct path — only
E_theta reveals the danger.

The intended contrast (paper §6.3)
------------------------------------
- A **reward-only** planner (ignores E_theta) drives along y ≈ 0 toward V's peak at
  [3, 0], accumulating high transition-model exposure through the corridor.
- A **Learned-Gate A-CLAP** planner raises effective uncertainty in the high-E_theta
  region and detoures to |y| ≈ 0.6–1.0, reaching the target with substantially lower
  transition-model exposure.
"""
import numpy as np

from .base_env import SimpleEnv
from ..fields import FieldSet, GaussianBumpField, ConstantField


def TransitionErrorEnv():
    start = np.array([-3.0, 0.0])
    target = np.array([3.0, 0.0])
    shortcut_center = np.array([0.0, 0.0])  # on the straight start→target path

    # Value field: Gaussian bump at the target — looks attractive along the direct path
    V = GaussianBumpField(target, width=1.5, amplitude=5.0)

    # Static fields look benign everywhere (including the direct y=0 path)
    U = ConstantField(0.05)
    C = ConstantField(0.0)
    O = ConstantField(0.0)  # noqa: E741

    # E_theta: anisotropic corridor straddling the direct y=0 path.
    # Wider in x (x_width=1.5) so the planner accumulates error across many steps;
    # very narrow in y (y_width=0.3) so detouring to |y| ≈ 0.6 drops exposure
    # dramatically — yet the direct path remains the natural greedy choice.
    _x_width = 1.5
    _y_width = 0.3
    _amp = 1.5

    class _EField:
        """Anisotropic Gaussian: high on y≈0 corridor, low off-axis."""
        def __call__(self, z):
            z = np.atleast_2d(np.asarray(z, dtype=float))
            d2 = (z[:, 0] / _x_width) ** 2 + (z[:, 1] / _y_width) ** 2
            return 0.05 + _amp * np.exp(-d2 / 2.0)

    E = _EField()

    fields = FieldSet(V=V, U=U, C=C, O=O, c=ConstantField(1.0), E_theta=E)
    env = SimpleEnv(
        fields=fields,
        start=start,
        target=np.atleast_2d(target),
        N_star=0.99,
        target_radius=0.6,
        shortcut_center=shortcut_center,
    )
    return env
