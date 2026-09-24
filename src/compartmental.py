"""M1 - Deterministic compartmental models (SIR, SEIR, SEIRD) as ODE systems.

One general right-hand side covers all three models. State vector::

    [S, E, I, R, D, V, C]

S susceptible, E exposed (infected, not yet infectious), I infectious,
R recovered, D dead, V vaccinated, C cumulative infections (a bookkeeping
counter used for incidence; it is not part of the population).

* ``SIR``   - E and D are unused (new infections go straight to I).
* ``SEIR``  - adds a latent period of mean ``1/sigma`` days.
* ``SEIRD`` - a fraction ``ifr`` of people leaving I die instead of recovering.

Optional extras for every model:

* vaccination: S -> V at per-capita rate ``nu`` (perfect, all-or-nothing vaccine)
* waning immunity: R -> S at rate ``omega`` and V -> S at rate ``omega_v``
* testing / isolation: extra removal from I at rate ``kappa`` (isolated people
  no longer transmit and are counted in R or D)

``beta``, ``nu`` and ``kappa`` may be constants or functions of time, which is
how interventions (M5) are expressed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Literal

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

ModelName = Literal["SIR", "SEIR", "SEIRD"]
MODELS: tuple[str, ...] = ("SIR", "SEIR", "SEIRD")
COMPARTMENTS: tuple[str, ...] = ("S", "E", "I", "R", "D", "V")
RateLike = float | Callable[[float], float]


@dataclass(frozen=True)
class ModelParams:
    """Rate parameters (all per day).

    Attributes:
        beta: transmission rate (contacts x transmission probability per day).
        gamma: removal rate from I (1 / mean infectious period).
        sigma: progression rate E -> I (1 / mean latent period). SEIR/SEIRD only.
        ifr: fraction of removals from I that are deaths. SEIRD only.
        nu: vaccination rate, fraction of S vaccinated per day.
        omega: waning rate of infection-acquired immunity (R -> S).
        omega_v: waning rate of vaccine-acquired immunity (V -> S).
        kappa: extra removal rate from I due to testing and isolation.
    """

    beta: float = 0.3
    gamma: float = 0.1
    sigma: float = 0.2
    ifr: float = 0.0
    nu: float = 0.0
    omega: float = 0.0
    omega_v: float = 0.0
    kappa: float = 0.0

    @property
    def r0(self) -> float:
        """Basic reproduction number beta / (gamma + kappa)."""
        return self.beta / (self.gamma + self.kappa)

    def with_(self, **changes: float) -> "ModelParams":
        """Return a copy with some fields changed."""
        return replace(self, **changes)


def _as_fn(value: RateLike) -> Callable[[float], float]:
    """Wrap a constant as a function of time (functions pass through)."""
    if callable(value):
        return value
    const = float(value)
    return lambda _t: const


def make_rhs(
    model: ModelName,
    params: ModelParams,
    beta_fn: RateLike | None = None,
    nu_fn: RateLike | None = None,
    kappa_fn: RateLike | None = None,
) -> Callable[[float, np.ndarray], np.ndarray]:
    """Build the ODE right-hand side ``f(t, y)`` for the chosen model.

    The flows out of every compartment appear exactly once as inflows elsewhere,
    so d/dt (S+E+I+R+D+V) = 0: total population is conserved by construction.
    """
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}, got {model!r}")
    beta_t = _as_fn(params.beta if beta_fn is None else beta_fn)
    nu_t = _as_fn(params.nu if nu_fn is None else nu_fn)
    kappa_t = _as_fn(params.kappa if kappa_fn is None else kappa_fn)
    has_latent = model in ("SEIR", "SEIRD")
    ifr = params.ifr if model == "SEIRD" else 0.0
    g, sigma, omega, omega_v = params.gamma, params.sigma, params.omega, params.omega_v

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        S, E, I, R, D, V, _C = y
        alive = S + E + I + R + V
        infection = beta_t(t) * S * I / alive
        vaccination = nu_t(t) * S
        removal = (g + kappa_t(t)) * I
        onset = sigma * E if has_latent else 0.0
        waning_r, waning_v = omega * R, omega_v * V

        dS = -infection - vaccination + waning_r + waning_v
        if has_latent:
            dE = infection - onset
            dI = onset - removal
        else:
            dE = 0.0
            dI = infection - removal
        dR = (1.0 - ifr) * removal - waning_r
        dD = ifr * removal
        dV = vaccination - waning_v
        return np.array([dS, dE, dI, dR, dD, dV, infection])

    return rhs


def initial_state(
    N: float, I0: float, E0: float = 0.0, R0_init: float = 0.0, V0: float = 0.0
) -> np.ndarray:
    """Initial state vector with everyone not otherwise assigned in S."""
    S0 = N - I0 - E0 - R0_init - V0
    if S0 < 0:
        raise ValueError("initial compartments exceed population size N")
    return np.array([S0, E0, I0, R0_init, 0.0, V0, I0 + E0], dtype=float)


def simulate(
    model: ModelName,
    params: ModelParams,
    N: float = 1e6,
    I0: float = 10.0,
    t_max: float = 180.0,
    E0: float = 0.0,
    V0: float = 0.0,
    beta_fn: RateLike | None = None,
    nu_fn: RateLike | None = None,
    kappa_fn: RateLike | None = None,
    t_eval: np.ndarray | None = None,
    rtol: float = 1e-8,
    atol: float = 1e-8,
) -> pd.DataFrame:
    """Integrate a compartmental model with ``scipy.integrate.solve_ivp``.

    Returns a DataFrame with columns ``t, S, E, I, R, D, V, C`` plus
    ``incidence`` (new infections per unit time between output points) and
    ``Reff`` (effective reproduction number).
    """
    rhs = make_rhs(model, params, beta_fn, nu_fn, kappa_fn)
    if t_eval is None:
        t_eval = np.linspace(0.0, t_max, int(round(t_max)) + 1)
    y0 = initial_state(N, I0, E0=E0, V0=V0)
    # max_step=0.5 day so step-function interventions are never stepped over.
    sol = solve_ivp(
        rhs, (float(t_eval[0]), float(t_eval[-1])), y0, t_eval=t_eval,
        method="RK45", rtol=rtol, atol=atol, max_step=0.5,
    )
    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")
    df = pd.DataFrame(sol.y.T, columns=[*COMPARTMENTS, "C"])
    df.insert(0, "t", sol.t)
    df["incidence"] = np.concatenate([[np.nan], np.diff(df["C"].to_numpy())])
    beta_t = _as_fn(params.beta if beta_fn is None else beta_fn)
    kappa_t = _as_fn(params.kappa if kappa_fn is None else kappa_fn)
    alive = df[["S", "E", "I", "R", "V"]].sum(axis=1).to_numpy()
    df["Reff"] = [
        beta_t(t) / (params.gamma + kappa_t(t)) * s / a
        for t, s, a in zip(df["t"], df["S"], alive)
    ]
    return df


def total_population(df: pd.DataFrame) -> np.ndarray:
    """Sum of all population compartments at each time point."""
    return df[list(COMPARTMENTS)].sum(axis=1).to_numpy()


# ---------------------------------------------------------------------------
# Analytical results used for validation
# ---------------------------------------------------------------------------

def final_size(r0: float, s0: float = 1.0) -> float:
    """Solve the classic SIR final-size equation for the attack fraction.

    With fractions s0 + i0 = 1 (no initial immunity) the fraction still
    susceptible at the end, s_inf, satisfies::

        ln(s0 / s_inf) = R0 * (1 - s_inf)

    The returned attack fraction is ``1 - s_inf`` (it includes the initially
    infected). The same equation holds for SEIR, since a latent period changes
    the timing but not the final size. Returns 1 - s0 (only the seeds) when
    R0 <= 1 and s0 -> 1.
    """
    if r0 <= 0:
        return 1.0 - s0

    def g(s_inf: float) -> float:
        return np.log(s0 / s_inf) - r0 * (1.0 - s_inf)

    # The non-trivial root lies below min(s0, 1/R0) when R0 > 1.
    upper = s0 - 1e-15 if r0 <= 1 else min(s0, 1.0 / r0)
    if g(upper) >= 0:  # no interior root: essentially no epidemic
        return 1.0 - s0
    s_inf = brentq(g, 1e-300, upper, xtol=1e-14)
    return 1.0 - s_inf


def sir_peak_prevalence(r0: float, s0: float = 1.0, i0: float = 0.0) -> float:
    """Analytical SIR peak infectious fraction.

    From the conserved quantity i + s - ln(s)/R0, evaluated at the peak where
    s = 1/R0::

        i_max = i0 + s0 - (1 + ln(R0 * s0)) / R0

    Only meaningful when R0 * s0 > 1.
    """
    return i0 + s0 - (1.0 + np.log(r0 * s0)) / r0


def herd_immunity_threshold(r0: float) -> float:
    """Fraction immune needed so that R_eff < 1: 1 - 1/R0 (0 if R0 <= 1)."""
    return max(0.0, 1.0 - 1.0 / r0)
