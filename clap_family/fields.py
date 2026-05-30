"""Scalar field abstractions for the conservative lapse (paper Eq. 1, 8).

A Field maps states z:(N,d) -> scalars:(N,). This is the seam where users plug in
their own neural value/uncertainty/OOD heads.
"""
from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Sequence

import numpy as np


class Field(Protocol):
    def __call__(self, z: np.ndarray) -> np.ndarray: ...


class CallableField:
    """Wrap any function f(z)->(N,) as a Field."""
    def __init__(self, fn: Callable[[np.ndarray], np.ndarray]):
        self.fn = fn

    def __call__(self, z):
        z = np.atleast_2d(np.asarray(z, dtype=float))
        return np.asarray(self.fn(z), dtype=float).reshape(z.shape[0])


class ConstantField:
    """Field returning a constant for every state."""
    def __init__(self, value: float):
        self.value = float(value)

    def __call__(self, z):
        z = np.atleast_2d(np.asarray(z, dtype=float))
        return np.full(z.shape[0], self.value)


class GaussianBumpField:
    """Isotropic Gaussian bump: amplitude*exp(-||z-center||^2/(2 width^2)) + baseline."""
    def __init__(self, center, width=1.0, amplitude=1.0, baseline=0.0):
        self.center = np.asarray(center, dtype=float)
        self.width = float(width)
        self.amplitude = float(amplitude)
        self.baseline = float(baseline)

    def __call__(self, z):
        z = np.atleast_2d(np.asarray(z, dtype=float))
        d2 = np.sum((z - self.center) ** 2, axis=-1)
        return self.amplitude * np.exp(-d2 / (2.0 * self.width ** 2)) + self.baseline


class EnsembleField:
    """Epistemic uncertainty as disagreement (std) across member fields."""
    def __init__(self, members: Sequence[Field]):
        self.members = list(members)

    def __call__(self, z):
        vals = np.stack([np.atleast_1d(m(z)) for m in self.members], axis=0)
        return np.std(vals, axis=0)


@dataclass
class FieldSet:
    """Bundle of fields the CLAP action needs (Eq. 1). E_theta optional (learned gate, Eq. 15)."""
    V: Field          # predicted value
    U: Field          # epistemic uncertainty
    C: Field          # safety cost
    O: Field          # OOD score  # noqa: E741
    c: Field          # latent speed limit c(z)
    E_theta: Optional[Field] = None   # transition-model error estimate
