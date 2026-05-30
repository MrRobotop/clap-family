"""ThreeRegionEnv (paper §6): safe high-lapse target, high-reward trap, unsafe/unreliable zone."""
import numpy as np

from .base_env import SimpleEnv
from ..fields import FieldSet, GaussianBumpField, ConstantField, CallableField


def ThreeRegionEnv():
    target = np.array([0.0, 0.0])
    trap = np.array([6.0, 0.0])
    unsafe = np.array([3.0, 3.0])
    # value: bumps at target and trap; trap is attractive but high cost/uncertainty/OOD
    V = CallableField(lambda z: (GaussianBumpField(target, 1.0, 5.0)(z)
                                 + GaussianBumpField(trap, 1.0, 8.0)(z)))
    U = GaussianBumpField(trap, 1.2, 4.0)
    C = CallableField(lambda z: (GaussianBumpField(trap, 1.2, 3.0)(z)
                                 + GaussianBumpField(unsafe, 1.0, 6.0)(z)))
    O = GaussianBumpField(unsafe, 1.0, 5.0)  # noqa: E741
    fields = FieldSet(V=V, U=U, C=C, O=O, c=ConstantField(1.0))
    return SimpleEnv(fields=fields, start=np.array([-3.0, 0.0]), target=np.atleast_2d(target),
                     N_star=0.99, target_radius=0.7,
                     trap_center=trap, trap_radius=0.9,
                     unsafe_center=unsafe, unsafe_radius=0.9)
