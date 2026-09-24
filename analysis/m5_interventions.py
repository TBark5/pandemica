"""M5 analysis: counterfactual scenarios and start-day x strength heatmaps."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from analysis.common import save_csv, save_json
from src.interventions import BASELINE, N_DEFAULT, Intervention, outcomes, run_scenario, sweep
from src.style import CATEGORICAL, HEAT_CMAP, savefig

START_DAYS = np.arange(0, 151, 10)
SWEEPS = {
    # kind: (strength grid, axis label, display multiplier, display unit)
    "lockdown": (np.round(np.arange(0, 0.91, 0.075), 3), "Contact reduction during 60-day lockdown", 100, "%"),
    "vaccination": (np.round(np.linspace(0, 0.01, 13), 5), "Vaccination rate (% of susceptibles per day)", 100, "%/day"),
    "isolation": (np.round(np.linspace(0, 0.3, 13), 4), "Extra isolation rate kappa (per day)", 1, "/day"),
}

SCENARIOS = {
    "No intervention": [],
    "Lockdown day 60, 60% for 60 d": [Intervention("lockdown", 60, 0.6, 60)],
    "Vaccination from day 60, 0.5%/day": [Intervention("vaccination", 60, 0.005)],
    "Test & isolate from day 30, 0.1/day": [Intervention("isolation", 30, 0.1)],
    "All three combined": [Intervention("lockdown", 60, 0.6, 60),
                           Intervention("vaccination", 60, 0.005),
                           Intervention("isolation", 30, 0.1)],
}


def counterfactuals() -> pd.DataFrame:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.6))
    rows = []
    for color, (name, ivs) in zip(CATEGORICAL, SCENARIOS.items()):
        df = run_scenario(ivs)
        ls = "--" if name == "No intervention" else "-"
        ax1.plot(df["t"], df["I"] / 1000, color=color, ls=ls, label=name)
        ax2.plot(df["t"], df["D"] / 1000, color=color, ls=ls, label=name)
        rows.append({"scenario": name, **outcomes(df)})
    ax1.axvspan(60, 120, color="grey", alpha=0.12, label="lockdown window")
    ax1.set_xlim(0, 400)
    ax2.set_xlim(0, 400)
    ax1.set_ylabel("Infectious (thousands)")
    ax2.set_ylabel("Cumulative deaths (thousands)")
    for ax in (ax1, ax2):
        ax.set_xlabel("Day")
    ax1.set_title("Counterfactual epidemic curves (SEIRD, R0 = 2.5, N = 1M)")
    ax2.set_title("Cumulative deaths")
    ax1.legend(fontsize=8.5)
    savefig(fig, "m5_counterfactuals")
    table = pd.DataFrame(rows)
    base = table.loc[table["scenario"] == "No intervention"].iloc[0]
    table["deaths_averted_pct"] = 100 * (1 - table["cumulative_deaths"] / base["cumulative_deaths"])
    table["peak_reduction_pct"] = 100 * (1 - table["peak_infectious"] / base["peak_infectious"])
    return table


def fig_heatmaps(grids: dict[str, pd.DataFrame]) -> None:
    """Rows = intervention type; columns = peak infections, cumulative deaths."""
    fig, axes = plt.subplots(3, 2, figsize=(13, 13))
    metrics = (("peak_infectious", "Peak infectious"), ("cumulative_deaths", "Cumulative deaths"))
    for row, (kind, grid) in enumerate(grids.items()):
        strengths, label, mult, _unit = SWEEPS[kind]
        for col, (metric, title) in enumerate(metrics):
            ax = axes[row, col]
            half = (strengths[1] - strengths[0]) / 2 * mult  # centre each cell on its value
            mat = grid.pivot(index="strength", columns="start_day", values=metric).to_numpy()
            im = ax.imshow(mat, origin="lower", aspect="auto", cmap=HEAT_CMAP,
                           norm=LogNorm(vmin=max(mat.min(), 1), vmax=mat.max()),
                           extent=[START_DAYS[0] - 5, START_DAYS[-1] + 5,
                                   strengths[0] * mult - half, strengths[-1] * mult + half])
            cb = fig.colorbar(im, ax=ax)
            cb.set_label(f"{title} (log scale)")
            ax.set_xlabel("Intervention start day")
            ax.set_ylabel(label, fontsize=9.5)
            ax.set_title(f"{kind.capitalize()}: {title.lower()}")
            ax.grid(False)
    fig.suptitle("When vs how strong: intervention start day x strength "
                 f"(no intervention: {grids['lockdown']['peak_infectious'].max():,.0f} peak infectious)",
                 y=1.0)
    fig.tight_layout()
    savefig(fig, "m5_intervention_heatmaps")


def main() -> dict:
    table = counterfactuals()
    save_csv("m5_counterfactuals", table)
    grids = {}
    for kind, (strengths, *_rest) in SWEEPS.items():
        grids[kind] = sweep(kind, START_DAYS, strengths)
        save_csv(f"m5_sweep_{kind}", grids[kind])
    fig_heatmaps(grids)

    lock = grids["lockdown"]
    strong = lock[lock["strength"] == lock["strength"].max()]
    best = lock.loc[lock["cumulative_deaths"].idxmin()]
    results = {
        "baseline": {"R0": BASELINE.r0, "N": N_DEFAULT, "ifr": BASELINE.ifr,
                     "latent_days": 1 / BASELINE.sigma, "infectious_days": 1 / BASELINE.gamma},
        "counterfactuals": table.set_index("scenario").to_dict(orient="index"),
        "lockdown_best_start_day_for_deaths": float(best["start_day"]),
        "lockdown_best_strength_for_deaths": float(best["strength"]),
        "lockdown_best_deaths": float(best["cumulative_deaths"]),
        "strongest_lockdown_deaths_by_start_day": dict(
            zip(strong["start_day"].astype(int).astype(str), strong["cumulative_deaths"])),
    }
    save_json("m5_summary", results)
    return results


if __name__ == "__main__":
    print(main())
