import numpy as np
from clap_family import AdaptiveDUCLAP, CLAP, DUCLAP
from clap_family.fields import FieldSet, ConstantField


def _fs():
    return FieldSet(V=ConstantField(1.0), U=ConstantField(0.0), C=ConstantField(0.0),
                    O=ConstantField(0.0), c=ConstantField(1.0))


def test_adaptive_gate_zero_is_clap():
    fs = _fs()
    z = np.cumsum(np.full((6, 2), 0.2), axis=0)
    a0 = AdaptiveDUCLAP(N_star=0.7, chi=3.0, omega=1.0, adaptive_gate=0.0, lambda_end=0.0)
    clap = CLAP(N_star=0.7, lambda_end=0.0)
    assert np.isclose(a0.action(z, fs), clap.action(z, fs))


def test_adaptive_gate_one_is_duclap():
    fs = _fs()
    z = np.cumsum(np.full((6, 2), 0.2), axis=0)
    a1 = AdaptiveDUCLAP(N_star=0.7, chi=3.0, omega=1.0, adaptive_gate=1.0, lambda_end=0.0)
    du = DUCLAP(N_star=0.7, chi=3.0, omega=1.0, lambda_end=0.0)
    assert np.isclose(a1.action(z, fs), du.action(z, fs))
