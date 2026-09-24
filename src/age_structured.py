"""Stretch goal - Age-structured SEIR model with a contact matrix.

Three age groups (children 0-19, adults 20-64, older adults 65+). ``C[i, j]`` is the
average number of daily contacts a person in group i has with people in group j.
Force of infection on group i::

    lambda_i = q * sum_j C[i, j] * I_j / N_j

where q is the probability of transmission per contact. Each group has its own
infection fatality ratio.

R0 comes from the **next-generation matrix** K[i, j] = q * C[i, j] * (N_i / N_j) / gamma
(expected infections in group i caused by one infectious person in group j):
R0 = largest eigenvalue of K. q is chosen so that R0 hits a target value.

The contact matrix and IFRs are ILLUSTRATIVE round numbers (in the spirit of
survey-based matrices such as POLYMOD, where children mix mostly with children and
adults have the most contacts overall). They are not real survey data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

GROUPS = ["0-19", "20-64", "65+"]
POPULATION = np.array([230_000, 590_000, 180_000], dtype=float)  # 1M people in total
# contacts per day by a person in row group with people in column group (illustrative)
CONTACTS = np.array([[9.0, 4.0, 0.8],
                     [1.56, 8.0, 1.2],
                     [1.02, 3.93, 2.5]])
IFR = np.array([0.0001, 0.003, 0.05])  # illustrative, rises steeply with age


def reciprocal(C: np.ndarray, N: np.ndarray) -> bool:
    """Contacts must balance: total i->j contacts equal total j->i contacts."""
    total = C * N[:, None]
    return bool(np.allclose(total, total.T, rtol=0.02))


def next_generation_matrix(q: float, C: np.ndarray, N: np.ndarray, gamma: float) -> np.ndarray:
    """K[i, j] = expected infections in group i from one infectious person in group j.

    One infectious person in j adds q * C[i, j] * S_i / N_j infections per day in group i
    (with S_i = N_i at the start) for 1/gamma days on average.
    """
    return q * C * (N[:, None] / N[None, :]) / gamma


def r0_from_ngm(q: float, C: np.ndarray, N: np.ndarray, gamma: float) -> float:
    """Spectral radius (largest absolute eigenvalue) of the next-generation matrix."""
    return float(np.max(np.abs(np.linalg.eigvals(next_generation_matrix(q, C, N, gamma)))))


def q_for_r0(r0: float, C: np.ndarray, N: np.ndarray, gamma: float) -> float:
    """Per-contact transmission probability that gives the target R0 (R0 is linear in q)."""
    return r0 / r0_from_ngm(1.0, C, N, gamma)


def simulate_age_seir(
    r0: float = 2.5, vaccinated: np.ndarray | None = None, C: np.ndarray = CONTACTS,
    N: np.ndarray = POPULATION, ifr: np.ndarray = IFR, sigma: float = 1 / 5,
    gamma: float = 1 / 7, seed_group: int = 1, seed_infected: float = 10.0,
    t_max: float = 730.0,
) -> pd.DataFrame:
    """Integrate the age-structured SEIR model; vaccinated people (per group) start immune.

    Returns one row per group with infections, attack rate and deaths.
    """
    k = len(N)
    q = q_for_r0(r0, C, N, gamma)
    V = np.zeros(k) if vaccinated is None else np.asarray(vaccinated, dtype=float)
    I0 = np.zeros(k)
    I0[seed_group] = seed_infected

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        S, E, I, R, Cum = y.reshape(5, k)
        lam = q * C @ (I / N)
        inf = lam * S
        return np.concatenate([-inf, inf - sigma * E, sigma * E - gamma * I, gamma * I, inf])

    y0 = np.concatenate([N - V - I0, np.zeros(k), I0, V, I0])
    sol = solve_ivp(rhs, (0, t_max), y0, method="RK45", rtol=1e-8, atol=1e-6)
    S, E, I, R, Cum = sol.y[:, -1].reshape(5, k)
    return pd.DataFrame({"group": GROUPS[:k] if k == 3 else list(range(k)), "population": N,
                         "vaccinated": V, "infections": Cum,
                         "attack_rate": Cum / N, "deaths": Cum * ifr})
