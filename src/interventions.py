"""M5 - Intervention simulator on top of the M1 SEIRD model.

Three interventions, each switched on at a start day:

* ``lockdown``    - transmission rate multiplied by ``1 - strength`` for
                    ``duration`` days, then back to normal.
* ``vaccination`` - susceptibles vaccinated at ``strength`` (fraction of S per day)
                    from the start day on.
* ``isolation``   - testing and isolation removes infectious people at extra rate
                    ``strength`` (per day) from the start day on.

Interventions combine: lockdown multiplies beta, isolation adds to kappa and
vaccination adds to nu. A counterfactual is simply the same model run with a
different list of interventions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import pandas as pd

from src.compartmental import ModelParams, simulate

Kind = Literal["lockdown", "vaccination", "isolation"]

# Baseline scenario: SEIRD, R0 = 2.5, 5-day latent period, 7-day infectious period.
BASELINE = ModelParams(beta=2.5 / 7, gamma=1 / 7, sigma=1 / 5, ifr=0.01)
N_DEFAULT = 1_000_000
I0_DEFAULT = 10
T_MAX = 730  # two years, so waves delayed by a lockdown are still counted


@dataclass(frozen=True)
class Intervention:
    """One intervention: what, when, how strong and (for lockdown) for how long."""

    kind: Kind
    start: float
    strength: float
    duration: float = 60.0

    def active(self, t: float) -> bool:
        if self.kind == "lockdown":
            return self.start <= t < self.start + self.duration
        return t >= self.start


def build_rate_functions(
    params: ModelParams, interventions: list[Intervention]
) -> tuple[Callable[[float], float], Callable[[float], float], Callable[[float], float]]:
    """Time-varying beta(t), nu(t) and kappa(t) for a list of interventions."""
    def beta_fn(t: float) -> float:
        b = params.beta
        for iv in interventions:
            if iv.kind == "lockdown" and iv.active(t):
                b *= 1.0 - iv.strength
        return b

    def nu_fn(t: float) -> float:
        return params.nu + sum(iv.strength for iv in interventions
                               if iv.kind == "vaccination" and iv.active(t))

    def kappa_fn(t: float) -> float:
        return params.kappa + sum(iv.strength for iv in interventions
                                  if iv.kind == "isolation" and iv.active(t))

    return beta_fn, nu_fn, kappa_fn


def run_scenario(
    interventions: list[Intervention],
    params: ModelParams = BASELINE,
    N: float = N_DEFAULT,
    I0: float = I0_DEFAULT,
    t_max: float = T_MAX,
) -> pd.DataFrame:
    """Simulate SEIRD with the given interventions (empty list = no intervention)."""
    beta_fn, nu_fn, kappa_fn = build_rate_functions(params, interventions)
    return simulate("SEIRD", params, N=N, I0=I0, t_max=t_max,
                    beta_fn=beta_fn, nu_fn=nu_fn, kappa_fn=kappa_fn)


def outcomes(df: pd.DataFrame) -> dict[str, float]:
    """Headline outcomes of one scenario."""
    N = float(df[["S", "E", "I", "R", "D", "V"]].iloc[0].sum())
    return {
        "peak_infectious": float(df["I"].max()),
        "peak_day": float(df["t"].iloc[df["I"].idxmax()]),
        "cumulative_deaths": float(df["D"].iloc[-1]),
        "cumulative_infections": float(df["C"].iloc[-1]),
        "attack_rate": float(df["C"].iloc[-1] / N),
        "vaccinated": float(df["V"].iloc[-1]),
    }


def sweep(
    kind: Kind,
    start_days: np.ndarray,
    strengths: np.ndarray,
    duration: float = 60.0,
    params: ModelParams = BASELINE,
    N: float = N_DEFAULT,
) -> pd.DataFrame:
    """Grid of start day x strength for one intervention type (long format)."""
    rows = []
    for start in start_days:
        for strength in strengths:
            iv = [Intervention(kind, float(start), float(strength), duration)]
            rows.append({"kind": kind, "start_day": float(start), "strength": float(strength),
                         **outcomes(run_scenario(iv, params, N=N))})
    return pd.DataFrame(rows)
