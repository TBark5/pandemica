"""Tests for M5 interventions."""

import numpy as np
import pytest

from src.compartmental import total_population
from src.interventions import (
    BASELINE,
    Intervention,
    build_rate_functions,
    outcomes,
    run_scenario,
    sweep,
)


def test_zero_strength_equals_no_intervention():
    base = outcomes(run_scenario([]))
    for kind in ("lockdown", "vaccination", "isolation"):
        o = outcomes(run_scenario([Intervention(kind, 20, 0.0)]))
        assert o["cumulative_deaths"] == pytest.approx(base["cumulative_deaths"], rel=1e-6)


def test_rate_functions_switch_on_and_off():
    ivs = [Intervention("lockdown", 10, 0.5, duration=20), Intervention("isolation", 5, 0.1),
           Intervention("vaccination", 15, 0.01)]
    beta, nu, kappa = build_rate_functions(BASELINE, ivs)
    assert beta(9.9) == BASELINE.beta
    assert beta(10) == pytest.approx(BASELINE.beta * 0.5)
    assert beta(30) == BASELINE.beta  # lockdown lifted
    assert kappa(4) == 0 and kappa(100) == pytest.approx(0.1)
    assert nu(14) == 0 and nu(15) == pytest.approx(0.01)


def test_population_conserved_with_interventions():
    df = run_scenario([Intervention("lockdown", 30, 0.7), Intervention("vaccination", 10, 0.01),
                       Intervention("isolation", 0, 0.05)])
    assert np.abs(total_population(df) - 1_000_000).max() < 1e-4


def test_isolation_above_threshold_stops_epidemic():
    """With kappa large enough that beta / (gamma + kappa) < 1, no epidemic occurs."""
    kappa = BASELINE.beta - BASELINE.gamma + 0.02  # R0 just below 1
    o = outcomes(run_scenario([Intervention("isolation", 0, kappa)]))
    assert o["peak_infectious"] <= 10 + 1e-6
    assert o["attack_rate"] < 1e-3


def test_stronger_permanent_measures_never_hurt():
    """Vaccination and isolation that stay on can only reduce deaths as they get stronger."""
    for kind, strengths in (("vaccination", [0, 0.002, 0.005, 0.01]),
                            ("isolation", [0, 0.05, 0.1, 0.2])):
        deaths = sweep(kind, np.array([30]), np.array(strengths))["cumulative_deaths"].to_numpy()
        assert np.all(np.diff(deaths) < 0)


def test_temporary_lockdown_too_early_only_delays():
    """An early temporary lockdown mostly postpones the epidemic: deaths barely change."""
    base = outcomes(run_scenario([]))
    early = outcomes(run_scenario([Intervention("lockdown", 10, 0.8, 60)]))
    assert early["peak_day"] > base["peak_day"] + 40
    assert early["cumulative_deaths"] == pytest.approx(base["cumulative_deaths"], rel=0.02)
