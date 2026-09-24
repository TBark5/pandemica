"""M6 analysis: LHS over SEIRD parameters, PRCC ranking, tornado plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.common import save_csv, save_json
from src.sensitivity import DEFAULT_RANGES, latin_hypercube, prcc, run_samples
from src.style import OKABE_ITO, savefig

N_SAMPLES = 800
OUTCOMES = {"cumulative_deaths": "Cumulative deaths", "peak_infectious": "Peak infectious",
            "peak_day": "Day of peak", "attack_rate": "Attack rate"}
LABELS = {"beta": "Transmission rate beta", "infectious_period": "Infectious period 1/gamma",
          "latent_period": "Latent period 1/sigma", "ifr": "Infection fatality ratio",
          "vaccination_rate": "Vaccination rate nu", "isolation_rate": "Isolation rate kappa",
          "initial_infected": "Initial infected I0"}


def tornado(ax, table: pd.DataFrame, title: str) -> None:
    """Horizontal PRCC bars, largest effect on top; faded bars are not significant."""
    t = table.iloc[::-1]
    colors = [OKABE_ITO["vermillion"] if v > 0 else OKABE_ITO["blue"] for v in t["prcc"]]
    alphas = [1.0 if p < 0.05 else 0.3 for p in t["p_value"]]
    y = np.arange(len(t))
    for yi, v, c, a in zip(y, t["prcc"], colors, alphas):
        ax.barh(yi, v, color=c, alpha=a, height=0.65)
        ax.text(v + (0.03 if v >= 0 else -0.03), yi, f"{v:+.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=9)
    ax.set_yticks(y, [LABELS[p] for p in t["parameter"]])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlim(-1.15, 1.15)
    ax.set_xlabel("PRCC")
    ax.set_title(title)
    ax.grid(axis="y", visible=False)


def main() -> dict:
    samples = latin_hypercube(DEFAULT_RANGES, N_SAMPLES, seed=7)
    data = run_samples(samples)
    save_csv("m6_lhs_samples_and_outcomes", data)
    X = data[list(DEFAULT_RANGES)]
    tables = {}
    for out in OUTCOMES:
        tables[out] = prcc(X, data[out])
        save_csv(f"m6_prcc_{out}", tables[out])

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    tornado(axes[0], tables["cumulative_deaths"], "Cumulative deaths")
    tornado(axes[1], tables["peak_infectious"], "Peak infectious")
    fig.suptitle(f"Which parameters drive the outcome? PRCC from {N_SAMPLES} Latin hypercube "
                 "samples (faded = p >= 0.05)", y=1.02)
    fig.tight_layout()
    savefig(fig, "m6_tornado")

    matrix = pd.DataFrame({OUTCOMES[o]: t.set_index("parameter")["prcc"] for o, t in tables.items()})
    matrix = matrix.loc[list(DEFAULT_RANGES)]
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    for (i, j), v in np.ndenumerate(matrix.to_numpy()):
        ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=9,
                color="white" if abs(v) > 0.6 else "black")
    ax.set_xticks(range(matrix.shape[1]), [c.replace(" ", "\n") for c in matrix.columns])
    ax.set_yticks(range(matrix.shape[0]), [LABELS[p] for p in matrix.index])
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="PRCC")
    ax.set_title("PRCC of every parameter with every outcome")
    savefig(fig, "m6_prcc_matrix")

    results = {"n_samples": N_SAMPLES, "ranges": DEFAULT_RANGES,
               "prcc": {o: t.set_index("parameter")[["prcc", "p_value"]].to_dict(orient="index")
                        for o, t in tables.items()},
               "top_driver": {o: t["parameter"].iloc[0] for o, t in tables.items()}}
    save_json("m6_summary", results)
    return results


if __name__ == "__main__":
    out = main()
    print(out["top_driver"])
