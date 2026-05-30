import pytest
from dataclasses import replace
from clap_family.params import CLAPParams


def test_defaults_are_valid():
    p = CLAPParams()
    assert p.tau > 0 and 0 < p.rho < 1 and p.cmax > 0
    assert p.beta == 1.0 and p.gamma == 0.5 and p.kappa == 0.1


def test_replace_overrides_one_field():
    p = replace(CLAPParams(), beta=2.0)
    assert p.beta == 2.0 and p.eta == 1.0


def test_invalid_temperature_rejected():
    with pytest.raises(ValueError):
        CLAPParams(tau=0.0)


def test_invalid_rho_rejected():
    with pytest.raises(ValueError):
        CLAPParams(rho=1.5)


def test_invalid_dt_rejected():
    with pytest.raises(ValueError):
        CLAPParams(dt=0.0)
