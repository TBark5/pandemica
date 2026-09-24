"""Tests for M7 metapopulation model."""

import numpy as np
import pytest

from src.compartmental import ModelParams, simulate
from src.metapopulation import (
    Geography,
    default_geography,
    gravity_flows,
    simulate_metapop,
)


def test_gravity_flows_symmetric_and_scaled():
    geo = default_geography()
    F = gravity_flows(geo, 0.01)
    np.testing.assert_allclose(F, F.T)
    assert np.all(np.diag(F) == 0)
    assert F.sum() == pytest.approx(0.01 * geo.population.sum())


def test_each_region_keeps_its_population():
    geo = default_geography()
    res = simulate_metapop(geo, gravity_flows(geo, 0.01), t_max=200)
    totals = res.S + res.E + res.I + res.R
    assert np.abs(totals - geo.population).max() / geo.population.min() < 1e-8


def test_no_travel_means_no_spread_and_matches_single_region_seir():
    geo = default_geography()
    res = simulate_metapop(geo, np.zeros((geo.n, geo.n)), beta=0.5, sigma=1 / 3, gamma=0.2,
                           t_max=200)
    assert res.I[:, 1:].max() == 0
    ode = simulate("SEIR", ModelParams(beta=0.5, gamma=0.2, sigma=1 / 3),
                   N=geo.population[0], I0=10, t_max=200)
    np.testing.assert_allclose(res.I[:, 0], ode["I"].to_numpy(), rtol=1e-4, atol=1e-2)


def test_travel_restriction_delays_arrival():
    geo = default_geography()
    F = gravity_flows(geo, 0.002)
    base = simulate_metapop(geo, F).summary()
    restricted = simulate_metapop(geo, 0.1 * F).summary()
    assert (restricted["arrival_day"].iloc[1:] > base["arrival_day"].iloc[1:]).all()


def test_unseeded_twin_region_lags_but_ends_with_same_attack_rate():
    geo = Geography(["A", "B"], np.array([1e5, 1e5]), np.array([[0.0, 0.0], [1.0, 0.0]]))
    F = gravity_flows(geo, 0.01)
    res = simulate_metapop(geo, F, seed_region=0, t_max=300)
    table = res.summary()
    assert table["arrival_day"].iloc[1] > table["arrival_day"].iloc[0]
    assert table["attack_rate"].iloc[1] == pytest.approx(table["attack_rate"].iloc[0], abs=0.005)
