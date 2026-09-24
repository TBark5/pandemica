"""M3 analysis: Gillespie replicates vs the deterministic curve, extinction, convergence."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.common import save_csv, save_json
from src.compartmental import ModelParams, final_size, simulate
from src.stochastic import extinction_probability_theory, run_ensemble
from src.style import CATEGORICAL, COMPARTMENT_COLORS, OKABE_ITO, savefig

N, I0, BETA, GAMMA = 1000, 2, 0.25, 0.1   # R0 = 2.5
N_REPS = 500
T_MAX = 200
SPAGHETTI_SHOWN = 150


def fig_spaghetti(ens, ode) -> None:
    """Replicate paths, their band, the ODE curve, and the final-size histogram."""
    major = ens.major()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.8), gridspec_kw={"width_ratios": [1.7, 1]})
    for r in range(min(SPAGHETTI_SHOWN, len(ens.I))):
        ax.plot(ens.grid, ens.I[r], lw=0.6, alpha=0.35,
                color=COMPARTMENT_COLORS["I"] if major[r] else OKABE_ITO["grey"])
    lo, med, hi = np.percentile(ens.I[major], [5, 50, 95], axis=0)
    ax.fill_between(ens.grid, lo, hi, color=COMPARTMENT_COLORS["I"], alpha=0.15,
                    label="90% band (major outbreaks)")
    ax.plot(ens.grid, med, color=COMPARTMENT_COLORS["I"], lw=2.2, label="median (major outbreaks)")
    ax.plot(ode["t"], ode["I"], color="black", lw=2.2, ls="--", label="deterministic ODE")
    ax.plot([], [], color=OKABE_ITO["grey"], lw=1, label="replicates that went extinct")
    ax.set_xlabel("Day")
    ax.set_ylabel("Number infectious")
    ax.set_title(f"Gillespie SIR: {SPAGHETTI_SHOWN} of {len(ens.I)} replicates "
                 f"(N = {ens.N}, I0 = {ens.I0}, R0 = {ens.r0:.1f})")
    ax.legend(loc="upper right")

    frac = ens.final_size / ens.N * 100
    bins = np.linspace(0, 100, 51)
    ax2.hist(frac[~major], bins=bins, color=OKABE_ITO["grey"], label="minor (extinct early)")
    ax2.hist(frac[major], bins=bins, color=COMPARTMENT_COLORS["I"], label="major outbreak")
    ax2.axvline(final_size(ens.r0, 1 - ens.I0 / ens.N) * 100, color="black", ls="--",
                label="final-size equation")
    ax2.set_yscale("log")
    ax2.set_xlabel("Final size (% of population infected)")
    ax2.set_ylabel("Replicates (log scale)")
    p_sim, p_th = ens.extinction_probability(), extinction_probability_theory(ens.r0, ens.I0)
    ax2.set_title(f"Extinction: {p_sim:.1%} simulated vs {p_th:.1%} theory")
    ax2.legend(loc="upper center")
    savefig(fig, "m3_stochastic_spaghetti")


def extinction_table() -> pd.DataFrame:
    """Simulated vs branching-theory extinction probability over R0 and I0."""
    rows = []
    for r0 in (1.5, 2.5, 4.0):
        for i0 in (1, 2, 3, 5):
            ens = run_ensemble(400, 2000, i0, r0 * GAMMA, GAMMA, 600, seed=int(r0 * 100 + i0))
            rows.append({"R0": r0, "I0": i0, "n_reps": 400, "N": 2000,
                         "extinction_simulated": ens.extinction_probability(),
                         "extinction_theory": extinction_probability_theory(r0, i0)})
    return pd.DataFrame(rows)


def fig_extinction(table: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for color, (r0, g) in zip(CATEGORICAL, table.groupby("R0")):
        ax.plot(g["I0"], g["extinction_theory"], color=color, ls="--")
        ax.scatter(g["I0"], g["extinction_simulated"], color=color, s=40, zorder=3,
                   label=f"R0 = {r0}")
    ax.plot([], [], color="grey", ls="--", label="theory (1/R0)^I0")
    ax.set_xlabel("Initial number infectious I0")
    ax.set_ylabel("Probability outbreak dies out")
    ax.set_title("Early extinction: Gillespie (dots) vs branching theory (lines)")
    ax.legend()
    savefig(fig, "m3_extinction_probability")


def convergence_table() -> pd.DataFrame:
    """RMS gap between ensemble-mean prevalence and the ODE, for growing N (I0 = 1% of N)."""
    grid = np.linspace(0, 150, 151)
    rows = []
    for n in (100, 300, 1_000, 3_000, 10_000, 30_000):
        i0 = n // 100
        ens = run_ensemble(40, n, i0, BETA, GAMMA, 150, grid=grid, seed=n)
        ode = simulate("SIR", ModelParams(beta=BETA, gamma=GAMMA), N=n, I0=i0, t_eval=grid)
        gap = np.sqrt(np.mean((ens.I.mean(axis=0) - ode["I"].to_numpy()) ** 2)) / n
        rows.append({"N": n, "I0": i0, "n_reps": 40, "rms_gap_fraction_of_N": gap})
    return pd.DataFrame(rows)


def fig_convergence(table: pd.DataFrame) -> float:
    slope = np.polyfit(np.log(table["N"]), np.log(table["rms_gap_fraction_of_N"]), 1)[0]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.loglog(table["N"], table["rms_gap_fraction_of_N"], "o-", color=CATEGORICAL[0],
              label=f"Gillespie mean vs ODE (slope {slope:.2f})")
    ref = table["rms_gap_fraction_of_N"].iloc[0] * np.sqrt(table["N"].iloc[0] / table["N"])
    ax.loglog(table["N"], ref, ls=":", color="grey", label="1/sqrt(N) reference")
    ax.set_xlabel("Population size N")
    ax.set_ylabel("RMS gap in I(t)/N")
    ax.set_title("Stochastic mean converges to the ODE as N grows")
    ax.legend()
    savefig(fig, "m3_convergence")
    return float(slope)


def main() -> dict:
    ens = run_ensemble(N_REPS, N, I0, BETA, GAMMA, T_MAX, seed=2024)
    ode = simulate("SIR", ModelParams(beta=BETA, gamma=GAMMA), N=N, I0=I0, t_max=T_MAX)
    fig_spaghetti(ens, ode)
    major = ens.major()
    peak = ens.I[major].max(axis=1)
    results = {
        "setup": {"N": N, "I0": I0, "beta": BETA, "gamma": GAMMA, "R0": BETA / GAMMA,
                  "n_reps": N_REPS, "major_outbreak_threshold_fraction": 0.1},
        "extinction_probability_simulated": ens.extinction_probability(),
        "extinction_probability_theory": extinction_probability_theory(BETA / GAMMA, I0),
        "major_outbreak_mean_attack_rate": float(ens.final_size[major].mean() / N),
        "final_size_equation_attack_rate": final_size(BETA / GAMMA, 1 - I0 / N),
        "major_outbreak_peak_I_median": float(np.median(peak)),
        "major_outbreak_peak_I_p5_p95": [float(v) for v in np.percentile(peak, [5, 95])],
        "deterministic_peak_I": float(ode["I"].max()),
    }
    ext = extinction_table()
    save_csv("m3_extinction", ext)
    fig_extinction(ext)
    conv = convergence_table()
    save_csv("m3_convergence", conv)
    results["convergence_loglog_slope"] = fig_convergence(conv)
    results["extinction_max_abs_error"] = float(
        (ext["extinction_simulated"] - ext["extinction_theory"]).abs().max())
    save_json("m3_summary", results)
    return results


if __name__ == "__main__":
    print(main())
