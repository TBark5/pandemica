"""Tests for M2: the fitting code must recover known parameters."""

import numpy as np
import pytest

from src.data import make_synthetic_outbreak
from src.fitting import (
    bootstrap_fit,
    fit_growth_rate,
    fit_sir,
    r0_from_growth_rate,
    sir_prevalence,
)


def test_fit_recovers_parameters_from_noise_free_data():
    t = np.arange(0, 21, dtype=float)
    obs = sir_prevalence(t, beta=1.2, gamma=0.3, I0=5, N=2000)
    fit = fit_sir(t, obs, N=2000)
    assert fit.beta == pytest.approx(1.2, rel=1e-3)
    assert fit.gamma == pytest.approx(0.3, rel=1e-3)
    assert fit.I0 == pytest.approx(5, rel=1e-2)
    assert fit.r0 == pytest.approx(4.0, rel=1e-3)


def test_fit_recovers_synthetic_parameters_within_5_percent():
    """Parameter-recovery check on the labelled SYNTHETIC dataset (Poisson noise)."""
    data = make_synthetic_outbreak()
    fit = fit_sir(data.t, data.observed, data.N)
    for name, est in (("beta", fit.beta), ("gamma", fit.gamma), ("R0", fit.r0)):
        assert est == pytest.approx(data.truth[name], rel=0.05), name


def test_bootstrap_ci_contains_point_estimate():
    data = make_synthetic_outbreak()
    fit = bootstrap_fit(data.t, data.observed, data.N, n_boot=20, seed=0)
    assert fit.boot.shape == (20, 3)
    lo, hi = fit.ci("R0")
    assert lo < fit.r0 < hi
    with pytest.raises(ValueError):
        fit_sir(data.t, data.observed, data.N).ci("R0")


def test_growth_rate_relation():
    assert r0_from_growth_rate(0.0, sigma=0.2, gamma=0.2) == pytest.approx(1.0)
    # SIR limit (sigma -> infinity): R0 = 1 + r / gamma
    assert r0_from_growth_rate(0.1, sigma=1e12, gamma=0.2) == pytest.approx(1.5)


def test_growth_rate_fit_on_exact_exponential():
    inc = 30 * np.exp(0.2 * np.arange(14))
    g = fit_growth_rate(inc, sigma=0.2, gamma=0.2, n_boot=50)
    assert g.r == pytest.approx(0.2, rel=1e-9)
    assert g.doubling_time == pytest.approx(np.log(2) / 0.2)
