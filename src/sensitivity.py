"""M6 - Global sensitivity analysis: Latin hypercube sampling + PRCC.

1. **Latin hypercube sampling (LHS).** Each parameter's range is cut into n equal
   slices, and each slice is sampled exactly once, with slices randomly paired across
   parameters. This covers the whole parameter space evenly with few model runs.
2. **Run the model** once per sample and record the outcomes.
3. **Partial rank correlation coefficient (PRCC).** Replace every value by its rank
   (so any monotonic relationship counts, not only a linear one). For parameter
   x_i, regress rank(x_i) and rank(y) on the ranks of all the *other* parameters and
   correlate the two sets of residuals. PRCC in [-1, 1] measures how strongly y
   rises (+) or falls (-) with x_i once the effect of the other parameters is
   removed (Marino et al., 2008, J. Theor. Biol.).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import qmc

from src.compartmental import ModelParams, simulate

# Parameter ranges for the SEIRD sensitivity analysis (per day unless stated).
DEFAULT_RANGES: dict[str, tuple[float, float]] = {
    "beta": (0.2, 0.6),                # transmission rate
    "infectious_period": (4.0, 10.0),  # days (1/gamma)
    "latent_period": (2.0, 7.0),       # days (1/sigma)
    "ifr": (0.002, 0.02),              # infection fatality ratio
    "vaccination_rate": (0.0, 0.005),  # fraction of S per day
    "isolation_rate": (0.0, 0.1),      # kappa, extra removal from I
    "initial_infected": (1.0, 100.0),
}


def latin_hypercube(ranges: dict[str, tuple[float, float]], n: int, seed: int = 0) -> pd.DataFrame:
    """``n`` LHS samples scaled to the given parameter ranges."""
    sampler = qmc.LatinHypercube(d=len(ranges), rng=np.random.default_rng(seed))
    unit = sampler.random(n)
    lo = np.array([r[0] for r in ranges.values()])
    hi = np.array([r[1] for r in ranges.values()])
    return pd.DataFrame(qmc.scale(unit, lo, hi), columns=list(ranges))


def seird_outcomes(row: pd.Series, N: float = 1e6, t_max: float = 730) -> dict[str, float]:
    """Run SEIRD for one parameter sample and return the outcomes of interest."""
    params = ModelParams(beta=row["beta"], gamma=1 / row["infectious_period"],
                         sigma=1 / row["latent_period"], ifr=row["ifr"],
                         nu=row["vaccination_rate"], kappa=row["isolation_rate"])
    df = simulate("SEIRD", params, N=N, I0=row["initial_infected"],
                  t_eval=np.arange(0, t_max + 1, 1.0))
    return {"peak_infectious": float(df["I"].max()),
            "peak_day": float(df["t"].iloc[df["I"].idxmax()]),
            "cumulative_deaths": float(df["D"].iloc[-1]),
            "attack_rate": float(df["C"].iloc[-1] / N)}


def run_samples(samples: pd.DataFrame,
                model: Callable[[pd.Series], dict[str, float]] = seird_outcomes) -> pd.DataFrame:
    """Evaluate ``model`` on every sample; returns samples and outcomes side by side."""
    outs = [model(row) for _, row in samples.iterrows()]
    return pd.concat([samples.reset_index(drop=True), pd.DataFrame(outs)], axis=1)


def prcc(X: pd.DataFrame, y: pd.Series | np.ndarray) -> pd.DataFrame:
    """PRCC of each column of ``X`` with ``y``, with two-sided p-values."""
    R = X.rank().to_numpy(dtype=float)
    ry = stats.rankdata(np.asarray(y, dtype=float))
    n, k = R.shape
    rows = []
    for i, name in enumerate(X.columns):
        others = np.column_stack([np.ones(n), np.delete(R, i, axis=1)])
        # residuals after removing the linear effect of the other (ranked) parameters
        res_x = R[:, i] - others @ np.linalg.lstsq(others, R[:, i], rcond=None)[0]
        res_y = ry - others @ np.linalg.lstsq(others, ry, rcond=None)[0]
        r = float(np.corrcoef(res_x, res_y)[0, 1])
        dof = n - 2 - (k - 1)
        t = r * np.sqrt(dof / max(1e-300, 1 - r**2))
        p = float(2 * stats.t.sf(abs(t), dof))
        rows.append({"parameter": name, "prcc": r, "p_value": p})
    out = pd.DataFrame(rows)
    return out.reindex(out["prcc"].abs().sort_values(ascending=False).index).reset_index(drop=True)
