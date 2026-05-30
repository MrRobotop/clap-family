import numpy as np
from clap_family import LearnedGateACLAP
from clap_family.fields import FieldSet, ConstantField, CallableField


def _fs(E):
    return FieldSet(V=ConstantField(1.0), U=ConstantField(0.0), C=ConstantField(0.0),
                    O=ConstantField(0.0), c=ConstantField(1.0),
                    E_theta=CallableField(lambda z: np.full(z.shape[0], E)))


def test_high_transition_error_raises_action():
    z = np.cumsum(np.full((6, 2), 0.2), axis=0)
    a_lo = LearnedGateACLAP(N_star=0.7, chi=3.0, omega=1.0, lambda_end=0.0).action(z, _fs(0.0))
    a_hi = LearnedGateACLAP(N_star=0.7, chi=3.0, omega=1.0, lambda_end=0.0).action(z, _fs(1.0))
    assert a_hi > a_lo


def test_missing_E_theta_raises():
    import pytest
    fs = FieldSet(V=ConstantField(1.0), U=ConstantField(0.0), C=ConstantField(0.0),
                  O=ConstantField(0.0), c=ConstantField(1.0))   # no E_theta
    z = np.zeros((5, 2))
    with pytest.raises(ValueError):
        LearnedGateACLAP(N_star=0.7).action(z, fs)
