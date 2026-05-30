import numpy as np
import pytest
from clap_family.lapse import sigmoid, conservative_lapse, estimate_N_star
from clap_family.fields import FieldSet, GaussianBumpField, ConstantField
from clap_family.admissible import AdmissibleSet
from clap_family.params import CLAPParams


def test_sigmoid_monotone_bounded():
    x = np.array([-10.0, 0.0, 10.0])
    s = sigmoid(x)
    assert np.all(np.diff(s) > 0) and np.all((s > 0) & (s < 1))
    assert np.isclose(s[1], 0.5)


def test_conservative_lapse_penalizes_uncertainty():
    safe = conservative_lapse(1.0, 0.0, 0.0, 0.0, 1, 1, 1, 1.0)
    risky = conservative_lapse(1.0, 5.0, 0.0, 0.0, 1, 1, 1, 1.0)
    assert safe > risky


def test_estimate_n_star_is_grid_max_over_admissible():
    fs = FieldSet(V=GaussianBumpField([0, 0], width=1.0, amplitude=4.0),
                  U=ConstantField(0.0), C=ConstantField(0.0),
                  O=ConstantField(0.0), c=ConstantField(1.0))
    adm = AdmissibleSet()  # all admissible
    grid = np.array([[0.0, 0.0], [2.0, 2.0], [5.0, 5.0]])
    N_star, targets = estimate_N_star(fs, adm, grid, CLAPParams())
    # max value at origin -> highest lapse there
    assert np.allclose(targets[0], [0.0, 0.0])
    nb_origin = conservative_lapse(4.0, 0, 0, 0, 1, 1, 1, 1.0)
    assert np.isclose(N_star, nb_origin)


def test_estimate_n_star_raises_on_empty_admissible():
    fs = FieldSet(V=ConstantField(1.0), U=ConstantField(10.0), C=ConstantField(0.0),
                  O=ConstantField(0.0), c=ConstantField(1.0))
    adm = AdmissibleSet(Umax=1.0)   # excludes all (U=10)
    with pytest.raises(ValueError):
        estimate_N_star(fs, adm, np.zeros((3, 2)), CLAPParams())


def test_conservative_lapse_rejects_nonpositive_tau():
    with pytest.raises(ValueError):
        conservative_lapse(1.0, 0, 0, 0, 1, 1, 1, 0.0)


def test_sigmoid_handles_extreme_values():
    assert np.isfinite(sigmoid(1e9))
    assert np.isfinite(sigmoid(-1e9))
