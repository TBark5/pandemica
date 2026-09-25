"""Validation tests for M3 (Gillespie) against deterministic and analytical results."""

import numpy as np
import pytest

from src.compartmental import ModelParams, final_size, simulate
from src.stochastic import (
    extinction_probability_theory,
    gillespie_sir,
    on_grid,
    run_ensemble,
)

BETA, GAMMA = 0.25, 0.1  # R0 = 2.5


def test_single_path_is_valid():
    rng = np.random.default_rng(0)
    times, S, I = gillespie_sir(500, 5, BETA, GAMMA, 1000, rng)
    assert np.all(np.diff(times) > 0)
    assert np.all((S >= 0) & (I >= 0) & (S + I <= 500))
    assert np.all(np.diff(S) <= 0)  # susceptibles can only decrease in SIR
    assert np.all(np.abs(np.diff(S)) + np.abs(np.diff(I)) >= 1)  # one event per step
    assert I[-1] == 0  # ran until extinction


def test_stochastic_mean_approaches_deterministic_as_n_grows():
    """Validation check: RMS gap between ensemble mean and ODE shrinks with N.

    Initial infected is a fixed 1% of N, the setting of Kurtz's law of large numbers.
    """
    grid = np.linspace(0, 150, 151)
    errors = []
    for N in (200, 2_000, 20_000):
        I0 = N // 100
        ens = run_ensemble(60, N, I0, BETA, GAMMA, 150, grid=grid, seed=N)
        ode = simulate("SIR", ModelParams(beta=BETA, gamma=GAMMA), N=N, I0=I0, t_eval=grid)
        errors.append(np.sqrt(np.mean((ens.I.mean(axis=0) - ode["I"].to_numpy()) ** 2)) / N)
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 2e-3


@pytest.mark.parametrize("I0", [1, 2, 3])
def test_extinction_probability_matches_branching_theory(I0):
    """P(extinction) ~ (1/R0)^I0 in a large population."""
    n = 1000
    ens = run_ensemble(n, 2_000, I0, BETA, GAMMA, 400, seed=10 + I0)
    p_theory = extinction_probability_theory(BETA / GAMMA, I0)
    se = np.sqrt(p_theory * (1 - p_theory) / n)
    assert abs(ens.extinction_probability() - p_theory) < 4 * se + 0.01


def test_major_outbreak_size_matches_final_size_equation():
    N = 5_000
    ens = run_ensemble(200, N, 5, BETA, GAMMA, 1000, seed=3)
    major = ens.final_size[ens.major()] / N
    assert major.mean() == pytest.approx(final_size(2.5, 1 - 5 / N), abs=0.01)


def test_no_major_outbreak_when_r0_below_one():
    ens = run_ensemble(300, 5_000, 5, 0.08, 0.1, 1000, seed=4)
    assert ens.extinction_probability() == 1.0
    assert extinction_probability_theory(0.8, 5) == 1.0


def test_same_seed_same_result():
    a = run_ensemble(20, 300, 2, BETA, GAMMA, 100, seed=7)
    b = run_ensemble(20, 300, 2, BETA, GAMMA, 100, seed=7)
    np.testing.assert_array_equal(a.I, b.I)


def test_on_grid_is_piecewise_constant():
    times = np.array([0.0, 1.0, 2.5])
    vals = np.array([10, 11, 12])
    np.testing.assert_array_equal(on_grid(times, vals, np.array([0, 0.5, 1, 2, 3])),
                                  [10, 10, 11, 11, 12])
