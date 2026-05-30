"""Learned-Gate A-CLAP (paper Eq. 15, 32).

The risk gate is learned from a transition-model-error estimate E_theta:
  r_theta(z) = sigmoid(alpha * (E_theta(z) - eps_model)).
Becomes conservative exactly where the world model is unreliable.
"""
from .a_clap import ACLAP
from ..lapse import sigmoid


class LearnedGateACLAP(ACLAP):
    def _resolve_risk(self, z_states, fields):
        if fields.E_theta is None:
            raise ValueError("LearnedGateACLAP requires FieldSet.E_theta (transition-error estimate)")
        p = self.params
        return sigmoid(p.alpha * (fields.E_theta(z_states) - p.eps_model))
