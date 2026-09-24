"""M2 extension - Bayesian parameter estimation with MCMC (emcee).

Model for the observed prevalence counts y_k::

    y_k ~ NegativeBinomial(mean = I(t_k; beta, gamma, I0), dispersion = k)

The negative binomial allows more spread than Poisson (variance = mean + mean^2 / k),
which real surveillance counts usually show. Priors are flat on the log scale
within wide bounds, for log(beta), log(gamma), log(I0) and log(k).

``emcee`` runs an ensemble of walkers (affine-invariant sampler, Goodman & Weare
2010). After discarding burn-in, the remaining samples approximate the posterior;
its percentiles give credible intervals, e.g. for R0 = beta / gamma.
"""

from __future__ import annotations

from dataclasses import dataclass

import emcee
import numpy as np
from scipy.integrate import odeint
from scipy.special import gammaln

from src.fitting import fit_sir

PARAM_NAMES = ("beta", "gamma", "I0", "k")
LOG_BOUNDS = np.log(np.array([[1e-2, 20.0], [1e-2, 10.0], [1e-2, 100.0], [0.1, 1e4]]))


def _sir_rhs(y: np.ndarray, _t: float, beta: float, gamma: float, N: float) -> list[float]:
    S, I = y
    infection = beta * S * I / N
    return [-infection, infection - gamma * I]


def sir_prevalence_fast(t: np.ndarray, beta: float, gamma: float, I0: float,
                        N: float) -> np.ndarray:
    """I(t) of the same SIR model as M1, from a 2-state system (R = N - S - I) solved with
    ``odeint`` (LSODA). About 8x faster than the general solver, which matters because
    MCMC evaluates the likelihood tens of thousands of times. A test checks it agrees
    with the M1 solver."""
    return odeint(_sir_rhs, [N - I0, I0], t, args=(beta, gamma, N), rtol=1e-8, atol=1e-8)[:, 1]


def nb_loglik(y: np.ndarray, mu: np.ndarray, k: float) -> float:
    """Negative-binomial log-likelihood with mean ``mu`` and dispersion ``k``."""
    mu = np.clip(mu, 1e-9, None)
    return float(np.sum(gammaln(y + k) - gammaln(k) - gammaln(y + 1)
                        + k * np.log(k / (k + mu)) + y * np.log(mu / (k + mu))))


def log_posterior(log_theta: np.ndarray, t: np.ndarray, y: np.ndarray, N: float) -> float:
    """Flat prior on log-parameters inside LOG_BOUNDS plus the NB log-likelihood."""
    if np.any(log_theta < LOG_BOUNDS[:, 0]) or np.any(log_theta > LOG_BOUNDS[:, 1]):
        return -np.inf
    beta, gamma, I0, k = np.exp(log_theta)
    mu = sir_prevalence_fast(t, beta, gamma, I0, N)
    if not np.all(np.isfinite(mu)):
        return -np.inf
    return nb_loglik(y, mu, k)


@dataclass
class McmcResult:
    samples: np.ndarray          # (n_samples, 4) in natural units: beta, gamma, I0, k
    acceptance_fraction: float
    autocorr_time: np.ndarray    # per parameter, in steps
    n_walkers: int
    n_steps: int
    burn_in: int

    @property
    def r0(self) -> np.ndarray:
        return self.samples[:, 0] / self.samples[:, 1]

    def interval(self, name: str, level: float = 0.95) -> tuple[float, float, float]:
        """(median, lower, upper) of the posterior for a parameter or ``R0``."""
        values = self.r0 if name == "R0" else self.samples[:, PARAM_NAMES.index(name)]
        a = (1 - level) / 2 * 100
        lo, med, hi = np.percentile(values, [a, 50, 100 - a])
        return float(med), float(lo), float(hi)


def run_mcmc(
    t: np.ndarray, y: np.ndarray, N: float, n_walkers: int = 24, n_steps: int = 4000,
    burn_in: int = 1000, thin: int = 5, seed: int = 0,
) -> McmcResult:
    """Sample the posterior, starting walkers near the least-squares estimate."""
    rng = np.random.default_rng(seed)
    ls = fit_sir(t, y, N)
    centre = np.log([ls.beta, ls.gamma, ls.I0, 20.0])
    start = centre + 0.02 * rng.standard_normal((n_walkers, 4))
    sampler = emcee.EnsembleSampler(n_walkers, 4, log_posterior, args=(t, y, N))
    sampler.random_state = np.random.RandomState(seed).get_state()  # reproducible moves
    sampler.run_mcmc(start, n_steps, progress=False)
    chain = sampler.get_chain(discard=burn_in, thin=thin, flat=True)
    tau = sampler.get_autocorr_time(discard=burn_in, quiet=True)
    return McmcResult(samples=np.exp(chain), acceptance_fraction=float(
        np.mean(sampler.acceptance_fraction)), autocorr_time=tau, n_walkers=n_walkers,
        n_steps=n_steps, burn_in=burn_in)
