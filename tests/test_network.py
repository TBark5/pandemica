"""Tests for M4 network epidemics."""

import networkx as nx
import numpy as np
import pytest

from src.network import (
    NETWORK_TYPES,
    build_network,
    choose_vaccinated,
    degree_stats,
    network_r0,
    simulate_network_sir,
)


@pytest.mark.parametrize("kind", NETWORK_TYPES)
def test_networks_have_requested_size_and_mean_degree(kind):
    G = build_network(kind, n=1000, mean_degree=8, seed=1)
    stats = degree_stats(G)
    assert stats["nodes"] == 1000
    assert stats["mean_degree"] == pytest.approx(8, rel=0.05)


def test_unknown_network_raises():
    with pytest.raises(ValueError):
        build_network("lattice")


def test_counts_are_conserved_and_infections_attributed():
    G = build_network("Erdos-Renyi", n=500, seed=2)
    run, _ = simulate_network_sir(G, 0.1, 0.2, n_seeds=5, days=150, seed=2)
    assert np.all(run.S + run.I + run.R == 500)
    assert np.all(np.diff(run.S) <= 0)
    # every infection except the 5 seeds has exactly one infector
    assert run.secondary.sum() == (run.infection_day >= 0).sum() - 5


def test_no_transmission_means_no_spread():
    G = build_network("Barabasi-Albert", n=500, seed=3)
    run, _ = simulate_network_sir(G, 0.0, 0.2, n_seeds=5, days=100, seed=3)
    assert run.attack_rate == pytest.approx(5 / 500)


def test_same_seed_same_epidemic():
    G = build_network("Watts-Strogatz", n=500, seed=4)
    a, _ = simulate_network_sir(G, 0.1, 0.2, seed=9)
    b, _ = simulate_network_sir(G, 0.1, 0.2, seed=9)
    np.testing.assert_array_equal(a.I, b.I)


def test_vaccinated_nodes_are_never_infected():
    G = build_network("Erdos-Renyi", n=500, seed=5)
    vacc = choose_vaccinated(G, 0.2, "random", seed=5)
    run, _ = simulate_network_sir(G, 0.2, 0.2, seed=5, vaccinated=vacc)
    assert len(vacc) == 100
    assert np.all(run.infection_day[vacc] == -1)


def test_network_r0_formula_on_regular_graph():
    """On a k-regular graph the excess degree is k - 1, and T matches the daily-step model.

    T = 1 - P(no transmission) where the infector survives n days without transmitting
    with probability (1-p)^n (1-q)^(n-1) q, summed over n >= 1.
    """
    tau, gamma = 0.2, 0.2
    p, q = 1 - np.exp(-tau), 1 - np.exp(-gamma)
    n = np.arange(1, 2000)
    T = 1 - np.sum((1 - p) ** n * (1 - q) ** (n - 1) * q)
    G = nx.random_regular_graph(6, 200, seed=0)
    assert network_r0(G, tau=tau, gamma=gamma) == pytest.approx(T * 5)


def test_transmissibility_tends_to_continuous_time_limit():
    """For small daily rates the discrete-time T approaches tau / (tau + gamma)."""
    G = nx.random_regular_graph(2, 100, seed=0)  # excess degree 1, so R0 = T
    assert network_r0(G, tau=1e-4, gamma=3e-4) == pytest.approx(0.25, rel=1e-3)


def test_degree_targeting_beats_random_on_scale_free_network():
    """Removing hubs is far more effective than random vaccination at the same coverage."""
    G = build_network("Barabasi-Albert", n=1000, seed=6)
    def mean_attack(strategy):
        return np.mean([
            simulate_network_sir(G, 0.06, 0.2, seed=s,
                                 vaccinated=choose_vaccinated(G, 0.1, strategy, seed=s))[0].attack_rate
            for s in range(10)])
    assert mean_attack("degree") < 0.5 * mean_attack("random")
