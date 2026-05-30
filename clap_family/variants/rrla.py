"""RRLA — Robust Relativistic Lapse-Action (paper Eq. 10, 29).

Realized lapse decreases when the trajectory moves quickly through a state:
deficit becomes (N* - N̄)/sqrt(1 - q).
"""
import numpy as np
from .base import BaseCLAP


class RRLA(BaseCLAP):
    def _lapse_deficit(self, N_bar, q):
        return (self.params.N_star - N_bar) / np.sqrt(1.0 - q)
