"""Theorem-aligned trajectory metrics (paper §6).

target_dwell  -> concentration (T5)        trap_dwell -> conservative target selection
unsafe_time   -> admissibility             transition_exposure -> learned gating
jerk_proxy    -> acceleration suppression (T6)
"""
import numpy as np

from .geometry import acceleration


def target_dwell(traj, env):
    """Fraction of states within the target region."""
    return float(np.mean(env.in_target(traj)))


def trap_dwell(traj, env):
    """Fraction of states inside the reward trap."""
    return float(np.mean(env.in_trap(traj)))


def unsafe_time(traj, env):
    """Fraction of states inside the unsafe region."""
    return float(np.mean(env.in_unsafe(traj)))


def transition_exposure(traj, env):
    """Mean transition-model-error estimate along the trajectory (requires E_theta)."""
    if env.fields.E_theta is None:
        raise ValueError("transition_exposure requires FieldSet.E_theta")
    return float(np.mean(env.fields.E_theta(np.atleast_2d(traj))))


def jerk_proxy(traj, dt=1.0):
    """Mean acceleration magnitude as a jerk/smoothness proxy."""
    traj = np.asarray(traj, dtype=float)
    if traj.shape[0] < 3:
        return 0.0
    a = acceleration(traj, dt)
    return float(np.mean(np.linalg.norm(a, axis=-1)))
