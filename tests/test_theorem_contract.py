"""The theorem stack as executable tests (paper §4). Each test names the theorem it guards."""
import numpy as np
import pytest

from clap_family import (CLAP, RRLA, DUCLAP, AdaptiveDUCLAP, ACLAP,
                         estimate_N_star, conservative_lapse)
from clap_family.admissible import AdmissibleSet
from clap_family.fields import FieldSet, ConstantField, GaussianBumpField

ALL_VARIANTS = [CLAP, RRLA, DUCLAP, AdaptiveDUCLAP, ACLAP]


def _admissible_fieldset():
    return FieldSet(V=GaussianBumpField([0, 0], 1.0, 5.0), U=ConstantField(0.1),
                    C=ConstantField(0.0), O=ConstantField(0.0), c=ConstantField(1.0),
                    E_theta=ConstantField(0.05))


def _grid():
    xs = np.linspace(-3, 3, 13)
    return np.array([[x, y] for x in xs for y in xs])


# ---- T1: target existence ----
def test_T1_target_existence_grid_max():
    fs = _admissible_fieldset()
    from clap_family.params import CLAPParams
    N_star, targets = estimate_N_star(fs, AdmissibleSet(), _grid(), CLAPParams())
    assert targets.shape[0] >= 1
    assert np.allclose(targets[0], [0.0, 0.0], atol=0.6)   # near the value peak


# ---- T2: nonnegativity (all variants, randomized admissible trajectories) ----
@pytest.mark.parametrize("cls", ALL_VARIANTS)
def test_T2_nonnegativity(cls):
    fs = _admissible_fieldset()
    from clap_family.params import CLAPParams
    N_star, targets = estimate_N_star(fs, AdmissibleSet(), _grid(), CLAPParams())
    rng = np.random.default_rng(0)
    for _ in range(20):
        z = rng.normal(0, 0.3, size=(8, 2))     # small admissible-region trajectories
        m = cls(N_star=N_star)
        assert m.action(z, fs, target_states=targets) >= -1e-9


# ---- T3: zero-loss dwell ----
def test_T3_zero_loss_dwell_only_at_target_at_rest():
    fs = _admissible_fieldset()
    nb0 = float(conservative_lapse(5.0, 0.1, 0, 0, 1, 1, 1, 1.0))   # value at origin
    m = CLAP(N_star=nb0, lambda_end=0.0)
    target = np.array([[0.0, 0.0]])
    rest = np.zeros((6, 2))                                  # at target, v=0, a=0
    assert np.allclose(m.running_cost(rest, fs, target), 0.0, atol=1e-6)
    moving = np.cumsum(np.full((6, 2), 0.2), axis=0)         # moving -> strictly positive
    assert np.all(m.running_cost(moving, fs, target) > 1e-6)


# ---- speed barrier stays finite under the rho margin ----
def test_speed_barrier_finite_under_margin():
    fs = _admissible_fieldset()
    m = CLAP(N_star=0.99, rho=1e-3)
    z = np.cumsum(np.full((6, 2), 0.95), axis=0)   # near speed limit (cmax default 1)
    assert np.all(np.isfinite(m.running_cost(z, fs)))


# ---- variant reduction: each variant collapses to base CLAP in its zero-limit ----
def test_duclap_reduces_to_clap():
    fs = _admissible_fieldset()
    z = np.cumsum(np.full((6, 2), 0.2), axis=0)
    assert np.isclose(DUCLAP(N_star=0.7, chi=0, omega=0).action(z, fs),
                      CLAP(N_star=0.7).action(z, fs))


def test_aclap_reduces_to_clap_zero_risk():
    fs = _admissible_fieldset()
    z = np.cumsum(np.full((6, 2), 0.2), axis=0)
    a = ACLAP(N_star=0.7, chi=3, omega=1, gamma1=1, kappa1=1, risk_field=ConstantField(0.0))
    assert np.isclose(a.action(z, fs), CLAP(N_star=0.7).action(z, fs))
