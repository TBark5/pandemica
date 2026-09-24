"""M2 - Parameter estimation by least squares, with bootstrap confidence intervals.

Main method: fit an SIR model's prevalence curve I(t) to observed prevalence.
Free parameters are the transmission rate ``beta``, the recovery rate ``gamma``
and the initial number infectious ``I0``; the population size N is known.
They are optimised on a log scale (which keeps them positive) with
``scipy.optimize.least_squares``.

Least squares is done on square-root counts: sqrt(model) vs sqrt(observed).
Count noise grows with the count (Poisson variance = mean), and the square root
makes the noise roughly equal in size at every time point, which is what
ordinary least squares and the residual bootstrap assume.

Uncertainty: a residual bootstrap. Residuals (on the sqrt scale) from the best
fit are resampled with replacement, added back onto the fitted curve to create
new pseudo-datasets, and each pseudo-dataset is refitted. The 2.5th and 97.5th
percentiles of the refitted values give 95% confidence intervals, including one
for R0 = beta/gamma.

Secondary method (for COVID-19 case counts): estimate the early exponential
growth rate r by log-linear regression and convert it to R0 with the SEIR
relation R0 = (1 + r/sigma)(1 + r/gamma) (Wallinga & Lipsitch, 2007).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from src.compartmental import ModelParams, initial_state, make_rhs


def sir_prevalence(t: np.ndarray, beta: float, gamma: float, I0: float, N: float) -> np.ndarray:
    """Infectious prevalence I(t) of the M1 SIR model at times ``t`` (t[0] = start)."""
    rhs = make_rhs("SIR", ModelParams(beta=beta, gamma=gamma))
    sol = solve_ivp(rhs, (t[0], t[-1]), initial_state(N, I0), t_eval=t,
                    method="RK45", rtol=1e-7, atol=1e-7)
    return sol.y[2]


@dataclass
class FitResult:
    """Point estimates, fitted curve and (optionally) bootstrap samples."""

    beta: float
    gamma: float
    I0: float
    N: float
    t: np.ndarray
    fitted: np.ndarray
    residuals: np.ndarray    # observed - fitted, on the sqrt scale
    rmse: float              # root-mean-square error on the original count scale
    boot: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))  # beta, gamma, I0

    @property
    def r0(self) -> float:
        return self.beta / self.gamma

    @property
    def infectious_period(self) -> float:
        return 1.0 / self.gamma

    def ci(self, name: str, level: float = 0.95) -> tuple[float, float]:
        """Percentile bootstrap confidence interval for beta, gamma, I0 or R0."""
        if len(self.boot) == 0:
            raise ValueError("no bootstrap samples; run bootstrap_fit first")
        cols = {"beta": self.boot[:, 0], "gamma": self.boot[:, 1], "I0": self.boot[:, 2],
                "R0": self.boot[:, 0] / self.boot[:, 1]}
        alpha = (1 - level) / 2 * 100
        lo, hi = np.percentile(cols[name], [alpha, 100 - alpha])
        return float(lo), float(hi)

    def summary(self) -> dict:
        """Dictionary of estimates (and CIs if bootstrapped) for the results files."""
        out = {"beta": float(self.beta), "gamma": float(self.gamma), "I0": float(self.I0),
               "R0": float(self.r0), "infectious_period_days": float(self.infectious_period),
               "rmse": float(self.rmse)}
        if len(self.boot):
            out["n_bootstrap"] = len(self.boot)
            for name in ("beta", "gamma", "I0", "R0"):
                out[f"{name}_ci95"] = list(self.ci(name))
        return out


def _sqrt(x: np.ndarray) -> np.ndarray:
    """Square root with negatives clipped to zero (variance-stabilising transform)."""
    return np.sqrt(np.clip(x, 0.0, None))


def _raw(x: np.ndarray) -> np.ndarray:
    """Identity transform (negatives clipped to zero)."""
    return np.clip(x, 0.0, None)


# scale name -> (forward transform, inverse transform)
SCALES = {"sqrt": (_sqrt, lambda y: np.clip(y, 0.0, None) ** 2), "raw": (_raw, _raw)}


def fit_sir(
    t: np.ndarray,
    observed: np.ndarray,
    N: float,
    guess: tuple[float, float, float] = (1.0, 0.3, 1.0),
    scale: str = "sqrt",
) -> FitResult:
    """Least-squares fit of SIR (beta, gamma, I0) to observed prevalence.

    ``scale="sqrt"`` (default) compares square roots of model and data; ``"raw"``
    compares counts directly (kept only to show why sqrt is needed). ``residuals`` in
    the result are on the chosen scale.
    """
    fwd, _inv = SCALES[scale]
    t = np.asarray(t, dtype=float)
    observed = np.asarray(observed, dtype=float)
    obs_s = fwd(observed)

    def residuals(log_theta: np.ndarray) -> np.ndarray:
        beta, gamma, I0 = np.exp(log_theta)
        return fwd(sir_prevalence(t, beta, gamma, I0, N)) - obs_s

    lower = np.log([1e-3, 1e-3, 1e-2])
    upper = np.log([20.0, 10.0, 0.5 * N])
    res = least_squares(residuals, np.log(guess), bounds=(lower, upper), method="trf")
    beta, gamma, I0 = np.exp(res.x)
    fitted = sir_prevalence(t, beta, gamma, I0, N)
    return FitResult(beta=beta, gamma=gamma, I0=I0, N=N, t=t, fitted=fitted,
                     residuals=obs_s - fwd(fitted),
                     rmse=float(np.sqrt(np.mean((observed - fitted) ** 2))))


def bootstrap_fit(
    t: np.ndarray, observed: np.ndarray, N: float, n_boot: int = 500, seed: int = 0,
    scale: str = "sqrt",
) -> FitResult:
    """Point fit plus a residual bootstrap (on the fitting scale) with ``n_boot`` refits."""
    fwd, inv = SCALES[scale]
    best = fit_sir(t, observed, N, scale=scale)
    rng = np.random.default_rng(seed)
    guess = (best.beta, best.gamma, best.I0)
    base = fwd(best.fitted)
    samples = []
    for _ in range(n_boot):
        pseudo = inv(base + rng.choice(best.residuals, size=len(t), replace=True))
        refit = fit_sir(t, pseudo, N, guess=guess, scale=scale)
        samples.append((refit.beta, refit.gamma, refit.I0))
    best.boot = np.array(samples)
    return best


def bootstrap_curves(fit: FitResult, t: np.ndarray) -> np.ndarray:
    """Prevalence curves for every bootstrap parameter set (rows) at times ``t``."""
    return np.array([sir_prevalence(t, b, g, i0, fit.N) for b, g, i0 in fit.boot])


# ---------------------------------------------------------------------------
# Early growth rate -> R0 (used on COVID-19 incidence)
# ---------------------------------------------------------------------------

@dataclass
class GrowthFit:
    """Exponential growth-rate fit of early incidence."""

    r: float
    r_ci95: tuple[float, float]
    r0: float
    r0_ci95: tuple[float, float]
    doubling_time: float
    n_days: int


def r0_from_growth_rate(r: float | np.ndarray, sigma: float, gamma: float) -> float | np.ndarray:
    """SEIR relation between growth rate and R0: (1 + r/sigma)(1 + r/gamma)."""
    return (1 + r / sigma) * (1 + r / gamma)


def fit_growth_rate(
    incidence: np.ndarray, sigma: float, gamma: float, n_boot: int = 1000, seed: int = 0,
    days: np.ndarray | None = None,
) -> GrowthFit:
    """Log-linear regression of incidence on time, with residual-bootstrap CIs.

    Pass raw (unsmoothed) daily counts: the residual bootstrap assumes independent
    residuals, and a rolling mean would make them strongly autocorrelated (CIs too
    narrow). ``days`` gives the day of each count when some days were dropped.
    """
    y = np.log(np.asarray(incidence, dtype=float))
    x = np.arange(len(y), dtype=float) if days is None else np.asarray(days, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    rng = np.random.default_rng(seed)
    boot = np.array([
        np.polyfit(x, slope * x + intercept + rng.choice(resid, len(x)), 1)[0]
        for _ in range(n_boot)
    ])
    r_lo, r_hi = np.percentile(boot, [2.5, 97.5])
    r0_boot = r0_from_growth_rate(boot, sigma, gamma)
    r0_lo, r0_hi = np.percentile(r0_boot, [2.5, 97.5])
    return GrowthFit(r=float(slope), r_ci95=(float(r_lo), float(r_hi)),
                     r0=float(r0_from_growth_rate(slope, sigma, gamma)),
                     r0_ci95=(float(r0_lo), float(r0_hi)),
                     doubling_time=float(np.log(2) / slope), n_days=len(y))
