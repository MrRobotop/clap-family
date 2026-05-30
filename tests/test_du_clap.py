import numpy as np
from clap_family import DUCLAP, CLAP
from clap_family.fields import FieldSet, ConstantField


def _fs():
    return FieldSet(V=ConstantField(1.0), U=ConstantField(0.0), C=ConstantField(0.0),
                    O=ConstantField(0.0), c=ConstantField(1.0))


def test_duclap_reduces_to_clap_when_chi_omega_zero():
    fs = _fs()
    z = np.cumsum(np.full((6, 2), 0.2), axis=0)
    du = DUCLAP(N_star=0.7, chi=0.0, omega=0.0, lambda_end=0.0)
    clap = CLAP(N_star=0.7, lambda_end=0.0)
    assert np.isclose(du.action(z, fs), clap.action(z, fs))


def test_duclap_motion_raises_uncertainty_and_action():
    fs = _fs()
    z = np.cumsum(np.full((6, 2), 0.2), axis=0)   # moving -> q>0
    du_off = DUCLAP(N_star=0.7, chi=0.0, omega=0.0, lambda_end=0.0)
    du_on = DUCLAP(N_star=0.7, chi=3.0, omega=1.0, lambda_end=0.0)
    # higher dynamic uncertainty lowers N̄ -> larger deficit -> larger action
    assert du_on.action(z, fs) > du_off.action(z, fs)
