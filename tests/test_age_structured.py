"""Tests for the age-structured SEIR model (stretch goal)."""

import numpy as np
import pytest

from src.age_structured import (
    CONTACTS,
    POPULATION,
    q_for_r0,
    r0_from_ngm,
    reciprocal,
    simulate_age_seir,
)
from src.compartmental import final_size


def test_default_contact_matrix_is_reciprocal():
    assert reciprocal(CONTACTS, POPULATION)


def test_q_gives_target_r0():
    q = q_for_r0(2.5, CONTACTS, POPULATION, gamma=1 / 7)
    assert r0_from_ngm(q, CONTACTS, POPULATION, gamma=1 / 7) == pytest.approx(2.5)


def test_proportionate_mixing_reduces_to_homogeneous_final_size():
    """If everyone mixes in proportion to group size, every group has the SIR final size."""
    N = np.array([300_000.0, 500_000.0, 200_000.0])
    C = 10 * np.tile(N / N.sum(), (3, 1))  # C[i, j] = c * N_j / N
    res = simulate_age_seir(2.0, C=C, N=N, ifr=np.zeros(3), seed_infected=1.0, t_max=1500)
    expected = final_size(2.0, 1 - 1 / N.sum())
    np.testing.assert_allclose(res["attack_rate"], expected, atol=2e-3)


def test_vaccinating_older_adults_minimises_deaths():
    doses = 100_000
    deaths = []
    for g in range(3):
        v = np.zeros(3)
        v[g] = doses
        deaths.append(simulate_age_seir(2.5, vaccinated=v)["deaths"].sum())
    assert np.argmin(deaths) == 2


def test_population_accounting():
    res = simulate_age_seir(2.5, vaccinated=np.array([10_000.0, 0, 0]))
    assert (res["infections"] <= res["population"] - res["vaccinated"] + 1e-6).all()
