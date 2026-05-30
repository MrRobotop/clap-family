import numpy as np
from clap_family import PhaseAdaptiveLGACLAP
from clap_family.fields import FieldSet, ConstantField, CallableField


def _fs(E=1.0):
    return FieldSet(V=ConstantField(1.0), U=ConstantField(0.0), C=ConstantField(0.0),
                    O=ConstantField(0.0), c=ConstantField(1.0),
                    E_theta=CallableField(lambda z: np.full(z.shape[0], E)))


def test_dwell_phase_is_more_conservative_than_access():
    # states near the target (within dwell_radius) get alpha_dwell; far states get alpha_access
    fs = _fs()
    near = np.zeros((6, 2))                       # at target origin -> dwell phase
    far = np.cumsum(np.full((6, 2), 0.2), axis=0) + 10.0   # far -> access phase
    m = PhaseAdaptiveLGACLAP(N_star=0.7, chi=3.0, omega=1.0, lambda_end=0.0,
                             dwell_radius=0.1, alpha_access=0.3, alpha_dwell=1.0)
    # near target, gate multiplier is larger -> uncertainty contribution larger per unit motion
    g_near = m._gate(near[:-1], np.zeros(5), fs, np.linspace(0, 1, 5))
    g_far = m._gate(far[:-1], np.full(5, 10.0), fs, np.linspace(0, 1, 5))
    assert np.all(g_near >= g_far)


def test_phase_adaptive_importable_and_runs():
    fs = _fs()
    z = np.cumsum(np.full((6, 2), 0.2), axis=0)
    J = PhaseAdaptiveLGACLAP(N_star=0.7, chi=1.0).action(z, fs, target_states=np.array([[0.0, 0.0]]))
    assert np.isfinite(J)


def test_phase_adaptive_without_target_is_all_dwell_phase():
    fs = _fs()
    z = np.cumsum(np.full((6, 2), 0.2), axis=0)
    m = PhaseAdaptiveLGACLAP(N_star=0.7, chi=1.0, alpha_access=0.3, alpha_dwell=1.0)
    # with no target_states, dist defaults to 0 -> dwell multiplier everywhere
    gate = m._gate(z[:-1], np.zeros(5), fs, np.linspace(0, 1, 5))
    base = m._resolve_risk(z[:-1], fs)
    assert np.allclose(gate, np.clip(m.params.alpha_dwell * base, 0.0, 1.0))
