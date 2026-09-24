"""Tests for the MCMC (stretch goal) fitting code."""

import numpy as np
import pytest

from src.bayes import log_posterior, nb_loglik, run_mcmc, sir_prevalence_fast
from src.data import make_synthetic_outbreak
from src.fitting import sir_prevalence


def test_fast_solver_matches_m1_solver():
    t = np.arange(0, 20, dtype=float)
    fast = sir_prevalence_fast(t, 1.6, 0.45, 2.0, 1000)
    ref = sir_prevalence(t, 1.6, 0.45, 2.0, 1000)
    np.testing.assert_allclose(fast, ref, atol=1e-3)


def test_negative_binomial_approaches_poisson_for_large_k():
    from scipy.stats import poisson

    y, mu = np.array([0, 3, 10, 25]), np.array([1.0, 4.0, 9.0, 30.0])
    assert nb_loglik(y, mu, 1e7) == pytest.approx(poisson.logpmf(y, mu).sum(), rel=1e-5)


def test_prior_bounds_reject_out_of_range():
    d = make_synthetic_outbreak()
    assert log_posterior(np.log([100.0, 0.4, 2.0, 10.0]), d.t, d.observed, d.N) == -np.inf


def test_short_chain_recovers_synthetic_r0():
    d = make_synthetic_outbreak()
    res = run_mcmc(d.t, d.observed, d.N, n_walkers=16, n_steps=600, burn_in=200, seed=0)
    med, lo, hi = res.interval("R0")
    assert lo < d.truth["R0"] < hi
    assert 0.2 < res.acceptance_fraction < 0.8
