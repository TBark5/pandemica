"""Validation tests for M1 against known analytical results."""

import numpy as np
import pytest

from src.compartmental import (
    MODELS,
    ModelParams,
    final_size,
    herd_immunity_threshold,
    simulate,
    sir_peak_prevalence,
    total_population,
)

N = 1_000_000
I0 = 10


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize(
    "extras",
    [
        {},
        {"nu": 0.01},
        {"omega": 1 / 90},
        {"nu": 0.005, "omega": 1 / 120, "omega_v": 1 / 180, "kappa": 0.05},
    ],
)
def test_population_is_conserved(model, extras):
    """S+E+I+R+D+V must equal N at every time point (floating-point tolerance)."""
    params = ModelParams(beta=0.35, gamma=0.1, sigma=0.25, ifr=0.02, **extras)
    df = simulate(model, params, N=N, I0=I0, t_max=365)
    rel_err = np.abs(total_population(df) - N).max() / N
    assert rel_err < 1e-10
    assert (df[["S", "E", "I", "R", "D", "V"]].to_numpy() > -1e-6).all()


@pytest.mark.parametrize("model", ["SIR", "SEIR"])
@pytest.mark.parametrize("r0", [1.3, 2.0, 3.0, 5.0])
def test_final_size_matches_final_size_equation(model, r0):
    """Simulated attack rate equals the root of ln(s0/s_inf) = R0 (1 - s_inf)."""
    gamma = 0.1
    params = ModelParams(beta=r0 * gamma, gamma=gamma, sigma=0.2)
    df = simulate(model, params, N=N, I0=I0, t_max=2000)
    assert df["I"].iloc[-1] < 1e-3 * N and df["E"].iloc[-1] < 1e-3 * N
    simulated = df["C"].iloc[-1] / N
    expected = final_size(r0, s0=1 - I0 / N)
    assert simulated == pytest.approx(expected, abs=1e-5)


def test_final_size_known_values():
    """Textbook values: R0=2 infects ~79.7% of the population."""
    assert final_size(2.0) == pytest.approx(0.7968121300, abs=1e-8)
    assert final_size(0.8) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("r0", [0.5, 0.9])
def test_no_epidemic_when_r0_below_one(model, r0):
    """With R0 < 1, prevalence never grows and almost nobody is infected."""
    gamma = 0.1
    params = ModelParams(beta=r0 * gamma, gamma=gamma, sigma=0.2, ifr=0.01)
    df = simulate(model, params, N=N, I0=100, E0=0, t_max=1000)
    infectious = (df["I"] + df["E"]).to_numpy()
    assert np.all(np.diff(infectious) <= 1e-9)
    # Total infections from 100 seeds is 100 / (1 - R0) in expectation.
    assert df["C"].iloc[-1] < 100 / (1 - r0) * 1.01
    assert df["C"].iloc[-1] / N < 1e-3


def test_sir_peak_matches_analytical():
    """Peak prevalence and S at the peak (= N/R0) match closed-form SIR results."""
    r0, gamma = 3.0, 0.1
    t = np.arange(0, 400, 0.01)
    df = simulate("SIR", ModelParams(beta=r0 * gamma, gamma=gamma), N=N, I0=I0, t_eval=t)
    peak = df["I"].idxmax()
    s0, i0 = 1 - I0 / N, I0 / N
    assert df["I"].max() / N == pytest.approx(sir_peak_prevalence(r0, s0, i0), abs=1e-6)
    # S falls ~3e-4 N per 0.01-day grid step near the peak, hence the tolerance.
    assert df["S"].iloc[peak] / N == pytest.approx(1 / r0, abs=5e-4)


def test_reff_crosses_one_at_peak():
    """R_eff = R0 S/N equals 1 exactly when SIR prevalence peaks."""
    t = np.arange(0, 300, 0.01)
    df = simulate("SIR", ModelParams(beta=0.25, gamma=0.1), N=N, I0=I0, t_eval=t)
    assert df["Reff"].iloc[df["I"].idxmax()] == pytest.approx(1.0, abs=1e-3)


def test_early_growth_rate_is_beta_minus_gamma():
    """In the early SIR phase I(t) grows like exp((beta - gamma) t)."""
    params = ModelParams(beta=0.4, gamma=0.1)
    df = simulate("SIR", params, N=1e9, I0=1, t_max=20)
    slope = np.polyfit(df["t"], np.log(df["I"]), 1)[0]
    assert slope == pytest.approx(0.3, rel=1e-3)


def test_vaccination_above_herd_immunity_prevents_epidemic():
    """Starting with more than 1 - 1/R0 vaccinated, the outbreak cannot grow."""
    r0 = 2.5
    v0 = (herd_immunity_threshold(r0) + 0.02) * N
    df = simulate("SIR", ModelParams(beta=r0 * 0.1, gamma=0.1), N=N, I0=I0, V0=v0, t_max=600)
    assert df["I"].max() <= I0 + 1e-6


def test_waning_immunity_gives_endemic_equilibrium():
    """With waning immunity SIRS settles at S* = N / R0."""
    r0 = 2.0
    params = ModelParams(beta=r0 * 0.1, gamma=0.1, omega=1 / 100)
    df = simulate("SIR", params, N=N, I0=I0, t_max=6000)
    assert df["S"].iloc[-1] / N == pytest.approx(1 / r0, abs=1e-3)
    assert df["I"].iloc[-1] > 1000


def test_seird_deaths_equal_ifr_times_removals():
    """In SEIRD, cumulative deaths are exactly ifr x everyone removed from I."""
    params = ModelParams(beta=0.3, gamma=0.1, sigma=0.2, ifr=0.015)
    df = simulate("SEIRD", params, N=N, I0=I0, t_max=800)
    removed = df["R"].iloc[-1] + df["D"].iloc[-1]
    assert df["D"].iloc[-1] == pytest.approx(0.015 * removed, rel=1e-6)


def test_bad_model_name_raises():
    with pytest.raises(ValueError):
        simulate("SIS", ModelParams())
