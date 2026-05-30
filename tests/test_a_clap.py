import numpy as np
from clap_family import ACLAP, CLAP
from clap_family.fields import FieldSet, ConstantField


def _fs():
    return FieldSet(V=ConstantField(1.0), U=ConstantField(0.0), C=ConstantField(0.0),
                    O=ConstantField(0.0), c=ConstantField(1.0))


def test_aclap_zero_risk_reduces_to_clap():
    fs = _fs()
    z = np.cumsum(np.full((6, 2), 0.2), axis=0)
    a = ACLAP(N_star=0.7, chi=3.0, omega=1.0, gamma1=1.0, kappa1=1.0, lambda_end=0.0,
              risk_field=ConstantField(0.0))
    clap = CLAP(N_star=0.7, lambda_end=0.0)
    assert np.isclose(a.action(z, fs), clap.action(z, fs))


def test_aclap_localizes_penalty_to_high_risk_states():
    fs = _fs()
    z = np.cumsum(np.full((6, 2), 0.2), axis=0)
    low = ACLAP(N_star=0.7, chi=3.0, omega=1.0, lambda_end=0.0, risk_field=ConstantField(0.1))
    high = ACLAP(N_star=0.7, chi=3.0, omega=1.0, lambda_end=0.0, risk_field=ConstantField(0.9))
    assert high.action(z, fs) > low.action(z, fs)
