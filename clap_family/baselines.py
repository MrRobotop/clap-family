"""Baselines for honest comparison (paper §6). RewardOnlyPlanner = the failure foil."""
import numpy as np

from .variants.base import BaseCLAP


class RewardOnlyPlanner(BaseCLAP):
    """Maximizes raw value only (no uncertainty/safety/OOD, no barriers). Demonstrates the
    reward-only failure mode: attraction to high raw-value but unsafe traps."""

    def running_cost(self, z, fields, target_states=None):
        states = np.asarray(z, dtype=float)[:-1]
        # minimize negative value => chase reward; ignore U, C, O, motion
        return -fields.V(states)

    def action(self, z, fields, target_states=None):
        return float(np.sum(self.running_cost(z, fields)) * self.params.dt)
