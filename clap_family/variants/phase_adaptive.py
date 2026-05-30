"""Phase-Adaptive Learned-Gate A-CLAP (paper Eq. 16, 33) — strongest research candidate.

The effective gate adds a phase multiplier: r_{theta,ph}(z,t) = alpha_ph(z,t) * r_theta(z),
with low conservatism during access and high conservatism near dwell / high-risk regions.
Phase is decided by distance to the target manifold: within dwell_radius -> dwell phase.
"""
import numpy as np
from .learned_gate import LearnedGateACLAP


class PhaseAdaptiveLGACLAP(LearnedGateACLAP):
    """Phase-adaptive learned-gate A-CLAP.

    Requires target_states to be passed to action()/running_cost(); if omitted,
    distance defaults to zero so every state is treated as dwell-phase (maximum conservatism).
    """

    def _gate(self, z_states, dist, fields, frac):
        base = self._resolve_risk(z_states, fields)            # r_theta(z)
        p = self.params
        mult = np.where(dist <= p.dwell_radius, p.alpha_dwell, p.alpha_access)
        return np.clip(mult * base, 0.0, 1.0)
