"""M1 analysis: compartmental model curves, phase portrait, R_eff, validation numbers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis.common import save_csv, save_json
from src.compartmental import (
    MODELS,
    ModelParams,
    final_size,
    simulate,
    sir_peak_prevalence,
    total_population,
)
from src.style import COMPARTMENT_COLORS, CATEGORICAL, savefig

N = 1_000_000
I0 = 10
BASE = ModelParams(beta=0.3, gamma=0.1, sigma=0.2, ifr=0.01)  # R0 = 3


def validation_table() -> pd.DataFrame:
    """Compare simulated final size / peak with the analytical formulas."""
    rows = []
    for r0 in (1.5, 2.0, 2.5, 3.0, 4.0):
        p = ModelParams(beta=r0 * 0.1, gamma=0.1, sigma=0.2)
        t = np.arange(0, 1500.0, 0.05)
        for model in ("SIR", "SEIR"):
            df = simulate(model, p, N=N, I0=I0, t_eval=t)
            row = {
                "model": model,
                "R0": r0,
                "attack_rate_simulated": df["C"].iloc[-1] / N,
                "attack_rate_final_size_eq": final_size(r0, 1 - I0 / N),
                "max_conservation_error_rel": np.abs(total_population(df) - N).max() / N,
            }
            if model == "SIR":
                row["peak_prevalence_simulated"] = df["I"].max() / N
                row["peak_prevalence_analytical"] = sir_peak_prevalence(r0, 1 - I0 / N, I0 / N)
            rows.append(row)
    return pd.DataFrame(rows)


def fig_epidemic_curves() -> dict:
    """Three-panel SIR / SEIR / SEIRD comparison with R0 = 3."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    summary = {}
    for ax, model in zip(axes, MODELS):
        df = simulate(model, BASE, N=N, I0=I0, t_max=240)
        cols = {"SIR": "SIR", "SEIR": "SEIR", "SEIRD": "SEIRD"}[model]
        for c in cols:
            ax.plot(df["t"], df[c] / N * 100, color=COMPARTMENT_COLORS[c], label=c)
        ax.set_title(f"{model} (R0 = {BASE.r0:.1f})")
        ax.set_xlabel("Day")
        summary[model] = {
            "peak_day": float(df["t"].iloc[df["I"].idxmax()]),
            "peak_infectious_pct": float(df["I"].max() / N * 100),
            "attack_rate_pct": float(df["C"].iloc[-1] / N * 100),
            "deaths": float(df["D"].iloc[-1]),
        }
        ax.legend(loc="center right")
    axes[0].set_ylabel("% of population")
    fig.suptitle("Deterministic compartmental models (N = 1,000,000)", y=1.02)
    savefig(fig, "m1_epidemic_curves")
    return summary


def fig_phase_portrait() -> None:
    """S-I phase plane for several R0, with the S = N/R0 peak line."""
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for color, r0 in zip(CATEGORICAL, (1.5, 2.0, 3.0, 5.0)):
        df = simulate("SIR", ModelParams(beta=r0 * 0.1, gamma=0.1), N=N, I0=I0, t_max=800)
        ax.plot(df["S"] / N, df["I"] / N, color=color, label=f"R0 = {r0}")
        ax.axvline(1 / r0, color=color, ls=":", lw=1.2)
    ax.set_xlabel("Susceptible fraction S/N")
    ax.set_ylabel("Infectious fraction I/N")
    ax.set_title("SIR phase portrait (S vs I)")
    ax.text(0.02, 0.97, "dotted lines: S/N = 1/R0,\nwhere prevalence peaks",
            transform=ax.transAxes, va="top", fontsize=9)
    ax.set_xlim(0, 1.02)
    ax.legend(loc="upper right")
    savefig(fig, "m1_phase_portrait")


def fig_reff() -> None:
    """R_eff(t) alongside prevalence: the peak happens exactly when R_eff = 1."""
    df = simulate("SIR", ModelParams(beta=0.25, gamma=0.1), N=N, I0=I0,
                  t_eval=np.arange(0, 250, 0.25))
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(df["t"], df["Reff"], color=CATEGORICAL[0], label="R_eff(t) = R0 S(t)/N")
    ax1.axhline(1, color="grey", ls="--", lw=1)
    ax1.set_ylabel("Effective reproduction number")
    ax1.set_xlabel("Day")
    ax2 = ax1.twinx()
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    ax2.fill_between(df["t"], df["I"] / N * 100, color=COMPARTMENT_COLORS["I"], alpha=0.25,
                     label="Infectious (% of N)")
    ax2.set_ylabel("Infectious, % of population")
    ax1.set_zorder(ax2.get_zorder() + 1)  # draw the R_eff line above the shading
    ax1.patch.set_visible(False)
    peak_t = df["t"].iloc[df["I"].idxmax()]
    ax1.axvline(peak_t, color="black", ls=":", lw=1)
    ax1.annotate("peak prevalence\n(R_eff crosses 1)", (peak_t, 1), xytext=(peak_t + 15, 1.8),
                 arrowprops={"arrowstyle": "->"}, fontsize=9)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")
    ax1.set_title("Effective reproduction number over time (SIR, R0 = 2.5)")
    savefig(fig, "m1_reff")


def fig_vaccination_waning() -> dict:
    """Effect of optional vaccination and waning-immunity compartments."""
    scenarios = {
        "SEIR baseline": BASE,
        "Vaccination 0.5%/day": BASE.with_(nu=0.005),
        "Waning immunity (1/180 d)": BASE.with_(omega=1 / 180),
        "Vaccination + waning": BASE.with_(nu=0.005, omega=1 / 180, omega_v=1 / 365),
    }
    fig, ax = plt.subplots(figsize=(8, 4.5))
    out = {}
    for color, (name, p) in zip(CATEGORICAL, scenarios.items()):
        df = simulate("SEIR", p, N=N, I0=I0, t_max=720)
        ax.plot(df["t"], df["I"] / N * 100, color=color, label=name)
        out[name] = {"peak_infectious_pct": float(df["I"].max() / N * 100),
                     # can exceed 100% when immunity wanes (reinfections)
                     "cumulative_infections_pct": float(df["C"].iloc[-1] / N * 100)}
    ax.set_xlabel("Day")
    ax.set_ylabel("Infectious, % of population")
    ax.set_title("Optional compartments: vaccination and waning immunity (SEIR, R0 = 3)")
    ax.legend()
    savefig(fig, "m1_vaccination_waning")
    return out


def main() -> dict:
    table = validation_table()
    save_csv("m1_validation", table)
    results = {
        "base_params": BASE.__dict__ | {"R0": BASE.r0, "N": N, "I0": I0},
        "epidemic_curves": fig_epidemic_curves(),
        "vaccination_waning": fig_vaccination_waning(),
        "max_abs_final_size_error": float(
            (table["attack_rate_simulated"] - table["attack_rate_final_size_eq"]).abs().max()),
        "max_abs_peak_error": float(
            (table["peak_prevalence_simulated"] - table["peak_prevalence_analytical"]).abs().max()),
        "max_conservation_error_rel": float(table["max_conservation_error_rel"].max()),
    }
    fig_phase_portrait()
    fig_reff()
    save_json("m1_summary", results)
    return results


if __name__ == "__main__":
    print(main())
