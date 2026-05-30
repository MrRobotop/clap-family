import numpy as np
from clap_family import RRLA, CLAP
from clap_family.fields import FieldSet, ConstantField


def _fs():
    return FieldSet(V=ConstantField(0.0), U=ConstantField(0.0), C=ConstantField(0.0),
                    O=ConstantField(0.0), c=ConstantField(1.0))


def test_rrla_inflates_deficit_with_speed():
    fs = _fs()
    # moving trajectory: q>0 so deficit/sqrt(1-q) > deficit
    z = np.cumsum(np.full((5, 2), 0.3), axis=0)
    rrla = RRLA(N_star=0.6, lambda_end=0.0, gamma=0.0, kappa=0.0)
    clap = CLAP(N_star=0.6, lambda_end=0.0, gamma=0.0, kappa=0.0)
    assert rrla.action(z, fs) > clap.action(z, fs)


def test_rrla_equals_clap_when_stationary():
    fs = _fs()
    z = np.zeros((5, 2))   # q=0 -> sqrt(1-q)=1
    rrla = RRLA(N_star=0.6, lambda_end=0.0)
    clap = CLAP(N_star=0.6, lambda_end=0.0)
    assert np.isclose(rrla.action(z, fs), clap.action(z, fs))
