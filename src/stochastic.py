"""M3 - Exact stochastic SIR simulation with the Gillespie algorithm.

The same SIR process as M1, but with whole people and random event timing.
Two event types can happen:

* infection  S + I -> 2I   at rate  beta * S * I / N
* recovery   I -> R        at rate  gamma * I

Gillespie's direct method: with total rate ``a = a_inf + a_rec``, the waiting
time to the next event is Exponential(a), and the event is an infection with
probability ``a_inf / a``. Repeating this gives exact sample paths of the
continuous-time Markov chain.

Useful analytical result (branching-process approximation, large N): starting
from ``I0`` infectious people, the probability that the outbreak goes extinct
before it takes off is ``(1 / R0) ** I0`` when R0 > 1 (and 1 when R0 <= 1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def gillespie_sir(
    N: int, I0: int, beta: float, gamma: float, t_max: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One exact SIR sample path.

    Returns event times and the S and I counts right after each event
    (the first entry is the initial state at t = 0).
    """
    S, I = N - I0, I0
    t = 0.0
    times, s_path, i_path = [0.0], [S], [I]
    block = 4096
    expo = rng.standard_exponential(block)
    unif = rng.random(block)
    k = 0
    while I > 0:
        a_inf = beta * S * I / N
        a_rec = gamma * I
        a_tot = a_inf + a_rec
        if k == block:  # refill pre-drawn random numbers (much faster than one at a time)
            expo = rng.standard_exponential(block)
            unif = rng.random(block)
            k = 0
        t += expo[k] / a_tot
        if t > t_max:
            break
        if unif[k] * a_tot < a_inf:
            S -= 1
            I += 1
        else:
            I -= 1
        k += 1
        times.append(t)
        s_path.append(S)
        i_path.append(I)
    return np.array(times), np.array(s_path), np.array(i_path)


def on_grid(times: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Value of a piecewise-constant path at each grid time."""
    idx = np.searchsorted(times, grid, side="right") - 1
    return values[np.clip(idx, 0, len(values) - 1)]


@dataclass
class Ensemble:
    """Many replicate runs sampled on a common time grid."""

    grid: np.ndarray
    I: np.ndarray            # (replicates, len(grid)) infectious counts
    S: np.ndarray            # (replicates, len(grid)) susceptible counts
    final_size: np.ndarray   # total ever infected per replicate (includes I0)
    N: int
    I0: int
    beta: float
    gamma: float

    @property
    def r0(self) -> float:
        return self.beta / self.gamma

    def major(self, threshold: float = 0.1) -> np.ndarray:
        """Boolean mask of major outbreaks (final size above ``threshold`` x N)."""
        return self.final_size > threshold * self.N

    def extinction_probability(self, threshold: float = 0.1) -> float:
        """Fraction of replicates that died out as minor outbreaks."""
        return float(np.mean(~self.major(threshold)))


def run_ensemble(
    n_reps: int,
    N: int,
    I0: int,
    beta: float,
    gamma: float,
    t_max: float,
    grid: np.ndarray | None = None,
    seed: int = 0,
) -> Ensemble:
    """Run ``n_reps`` independent Gillespie replicates with a seeded generator."""
    grid = np.linspace(0, t_max, int(t_max) + 1) if grid is None else grid
    rng = np.random.default_rng(seed)
    I_mat = np.empty((n_reps, len(grid)))
    S_mat = np.empty((n_reps, len(grid)))
    finals = np.empty(n_reps)
    for r in range(n_reps):
        times, s_path, i_path = gillespie_sir(N, I0, beta, gamma, t_max, rng)
        I_mat[r] = on_grid(times, i_path, grid)
        S_mat[r] = on_grid(times, s_path, grid)
        finals[r] = N - s_path[-1]
    return Ensemble(grid=grid, I=I_mat, S=S_mat, final_size=finals, N=N, I0=I0,
                    beta=beta, gamma=gamma)


def extinction_probability_theory(r0: float, I0: int) -> float:
    """Branching-process extinction probability (1/R0)^I0, or 1 if R0 <= 1."""
    return 1.0 if r0 <= 1 else (1.0 / r0) ** I0
