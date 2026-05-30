"""HardEnv (paper §6.2): narrow passage, reward trap, decoy plateau — reward-only collapses here."""
import numpy as np

from .base_env import SimpleEnv
from ..fields import FieldSet, GaussianBumpField, ConstantField, CallableField


def HardEnv():
    target = np.array([0.0, 0.0])
    trap = np.array([5.0, 0.0])
    V = CallableField(lambda z: (GaussianBumpField(target, 0.8, 5.0)(z)
                                 + GaussianBumpField(trap, 0.8, 9.0)(z)))
    U = GaussianBumpField(trap, 1.0, 5.0)
    C = CallableField(lambda z: (GaussianBumpField(trap, 1.0, 6.0)(z)
                                 + GaussianBumpField([2.5, 2.5], 0.7, 7.0)(z)))   # passage walls
    O = GaussianBumpField(trap, 1.0, 4.0)  # noqa: E741
    fields = FieldSet(V=V, U=U, C=C, O=O, c=ConstantField(1.0))
    return SimpleEnv(fields=fields, start=np.array([-4.0, 0.0]), target=np.atleast_2d(target),
                     N_star=0.99, target_radius=0.6,
                     trap_center=trap, trap_radius=0.8,
                     unsafe_center=[2.5, 2.5], unsafe_radius=0.8)  # unsafe region = the narrow passage walls (metrics), distinct from the trap
