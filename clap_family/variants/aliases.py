"""Snake-case functional wrappers for one-liner parity with the paper's Listing 1.

Each instantiates the corresponding class and returns the scalar action. The class API
is canonical; these exist so `from clap_family import du_clap` works as written.
"""
from .clap import CLAP
from .rrla import RRLA
from .du_clap import DUCLAP
from .adaptive_du import AdaptiveDUCLAP
from .a_clap import ACLAP
from .learned_gate import LearnedGateACLAP
from .phase_adaptive import PhaseAdaptiveLGACLAP


def _make(cls):
    def _fn(z, fields, params=None, target_states=None, **kwargs):
        return cls(params=params, **kwargs).action(z, fields, target_states=target_states)
    _fn.__name__ = cls.__name__.lower()
    _fn.__doc__ = f"Functional form of {cls.__name__}.action(z, fields). See class docstring."
    return _fn


clap = _make(CLAP)
rrla = _make(RRLA)
du_clap = _make(DUCLAP)
adaptive_du_clap = _make(AdaptiveDUCLAP)
a_clap = _make(ACLAP)
learned_gate_a_clap = _make(LearnedGateACLAP)
phase_adaptive_lg_a_clap = _make(PhaseAdaptiveLGACLAP)
