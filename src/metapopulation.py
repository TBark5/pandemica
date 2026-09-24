"""M7 - Metapopulation (spatial) SEIR model: regions connected by travel.

Each region i runs its own SEIR epidemic with well-mixed contacts inside the
region. People also move between regions: ``F[i, j]`` is the number of people
travelling from i to j per day, so a person in i leaves for j at per-capita rate
``m[i, j] = F[i, j] / P[i]``. For every compartment X in {S, E, I, R}::

    dX_i/dt = (local SEIR terms) + sum_j m[j, i] X_j - sum_j m[i, j] X_i

Travel flows come from a gravity model, F[i, j] ~ P_i P_j / d_ij^2, the simplest
widely used mobility model. F is symmetric, so every region keeps its
population constant over time (as many people arrive as leave).

The regions are fictional (a made-up map), which keeps the model clearly
illustrative.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


@dataclass
class Geography:
    """Region names, populations and map coordinates (arbitrary distance units)."""

    names: list[str]
    population: np.ndarray
    xy: np.ndarray

    @property
    def n(self) -> int:
        return len(self.names)

    def distances(self) -> np.ndarray:
        diff = self.xy[:, None, :] - self.xy[None, :, :]
        return np.sqrt((diff**2).sum(axis=-1))


def default_geography() -> Geography:
    """Eight fictional regions: a large capital, mid-sized cities and small towns."""
    names = ["Capital", "Northport", "Eastvale", "Southbay", "Westfield",
             "Highland", "Lakeside", "Farmstead"]
    population = np.array([2_000_000, 800_000, 600_000, 700_000, 500_000,
                           250_000, 300_000, 150_000], dtype=float)
    xy = np.array([[0.0, 0.0], [0.6, 3.6], [3.8, 0.9], [0.9, -3.4], [-3.6, -0.3],
                   [-1.8, 6.0], [5.8, 3.6], [-5.6, -4.2]])
    return Geography(names, population, xy)


def gravity_flows(geo: Geography, daily_travel_fraction: float = 0.005) -> np.ndarray:
    """Symmetric gravity-model flows F (people/day), zero diagonal.

    Scaled so that ``daily_travel_fraction`` of the total population travels each day.
    """
    d = geo.distances()
    np.fill_diagonal(d, np.inf)
    F = np.outer(geo.population, geo.population) / d**2
    F *= daily_travel_fraction * geo.population.sum() / F.sum()
    return F


@dataclass
class MetapopResult:
    t: np.ndarray
    S: np.ndarray   # (time, region)
    E: np.ndarray
    I: np.ndarray
    R: np.ndarray
    C: np.ndarray   # cumulative infections
    geo: Geography

    def prevalence(self) -> np.ndarray:
        """Infectious fraction per region over time."""
        return self.I / self.geo.population

    def summary(self, arrival_threshold: float = 1e-4) -> pd.DataFrame:
        """Arrival day (prevalence first above threshold), peak day, attack rate per region.

        Attack rate is the share of the people in the region at the end who were ever
        infected, (E + I + R) / P. (Counting infections by where they happened would
        credit the first-hit region with infecting its visitors.)
        """
        prev = self.prevalence()
        rows = []
        for k, name in enumerate(self.geo.names):
            above = np.flatnonzero(prev[:, k] >= arrival_threshold)
            rows.append({"region": name, "population": int(self.geo.population[k]),
                         "arrival_day": float(self.t[above[0]]) if len(above) else np.nan,
                         "peak_day": float(self.t[np.argmax(prev[:, k])]),
                         "peak_prevalence": float(prev[:, k].max()),
                         "attack_rate": float((self.E[-1, k] + self.I[-1, k] + self.R[-1, k])
                                              / self.geo.population[k])})
        return pd.DataFrame(rows)


def simulate_metapop(
    geo: Geography,
    F: np.ndarray,
    beta: float = 0.5,
    sigma: float = 1 / 3,
    gamma: float = 1 / 5,
    seed_region: int = 0,
    seed_infected: float = 10.0,
    t_max: float = 300.0,
) -> MetapopResult:
    """Integrate the coupled SEIR system with ``solve_ivp`` (daily output)."""
    n, P = geo.n, geo.population
    m = F / P[:, None]                  # per-capita travel rate i -> j
    out_rate = m.sum(axis=1)

    def move(X: np.ndarray) -> np.ndarray:
        return m.T @ X - out_rate * X   # arrivals minus departures

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        S, E, I, R, _C = y.reshape(5, n)
        Nloc = S + E + I + R
        inf = beta * S * I / Nloc
        return np.concatenate([-inf + move(S), inf - sigma * E + move(E),
                               sigma * E - gamma * I + move(I), gamma * I + move(R), inf])

    I0 = np.zeros(n)
    I0[seed_region] = seed_infected
    y0 = np.concatenate([P - I0, np.zeros(n), I0, np.zeros(n), I0])
    t = np.arange(0, t_max + 1, 1.0)
    sol = solve_ivp(rhs, (0, t_max), y0, t_eval=t, method="RK45", rtol=1e-8, atol=1e-6)
    if not sol.success:
        raise RuntimeError(sol.message)
    S, E, I, R, C = sol.y.reshape(5, n, -1).transpose(0, 2, 1)
    return MetapopResult(t=sol.t, S=S, E=E, I=I, R=R, C=C, geo=geo)
