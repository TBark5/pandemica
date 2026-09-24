"""Stretch goal: who should be vaccinated first? Age-structured SEIR with a contact matrix."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.common import save_csv, save_json
from src.age_structured import (
    CONTACTS,
    GROUPS,
    IFR,
    POPULATION,
    next_generation_matrix,
    q_for_r0,
    simulate_age_seir,
)
from src.style import CATEGORICAL, SEQUENTIAL_CMAP, savefig

R0 = 2.5
GAMMA = 1 / 7
DOSES = (0, 50_000, 100_000, 150_000, 200_000, 300_000)
PRIORITY = {"children first": [0, 1, 2], "adults 20-64 first": [1, 0, 2],
            "65+ first": [2, 1, 0], "everyone equally": None}


def allocate(doses: float, order: list[int] | None) -> np.ndarray:
    """Fill groups in priority order (spilling over when a group is fully vaccinated)."""
    if order is None:
        return doses * POPULATION / POPULATION.sum()
    v, left = np.zeros(len(POPULATION)), float(doses)
    for g in order:
        v[g] = min(left, POPULATION[g] * 0.999)
        left -= v[g]
    return v


def run() -> pd.DataFrame:
    rows = []
    for strategy, order in PRIORITY.items():
        for d in DOSES:
            res = simulate_age_seir(R0, vaccinated=allocate(d, order), gamma=GAMMA)
            rows.append({"strategy": strategy, "doses": d,
                         "infections": res["infections"].sum(), "deaths": res["deaths"].sum()})
    return pd.DataFrame(rows)


def figure(table: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), gridspec_kw={"width_ratios": [0.9, 1, 1]})
    ax = axes[0]
    im = ax.imshow(CONTACTS, cmap=SEQUENTIAL_CMAP)
    for (i, j), v in np.ndenumerate(CONTACTS):
        ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                color="white" if v < 6 else "black", fontsize=10)
    ax.set_xticks(range(3), GROUPS)
    ax.set_yticks(range(3), GROUPS)
    ax.set_xlabel("Contact's age group")
    ax.set_ylabel("Person's age group")
    ax.set_title("Daily contacts (illustrative matrix)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8)
    for ax, metric, label in ((axes[1], "infections", "Total infections"),
                              (axes[2], "deaths", "Total deaths")):
        for color, (strategy, g) in zip(CATEGORICAL, table.groupby("strategy", sort=False)):
            ax.plot(g["doses"] / 1000, g[metric] / 1000, "o-", color=color, label=strategy)
        ax.set_xlabel("Vaccine doses (thousands, population 1M)")
        ax.set_ylabel(f"{label} (thousands)")
        ax.set_title(f"{label} by vaccination priority")
    axes[1].legend(fontsize=9)
    fig.suptitle(f"Age-structured SEIR (R0 = {R0}): prioritising 65+ gives the fewest deaths at "
                 "every dose level; the best choice for infections depends on supply", y=1.03)
    savefig(fig, "age_vaccination_priority")


def main() -> dict:
    table = run()
    save_csv("age_vaccination", table)
    figure(table)
    q = q_for_r0(R0, CONTACTS, POPULATION, GAMMA)
    at = table[table["doses"] == 100_000].set_index("strategy")
    results = {
        "R0": R0, "per_contact_transmission_prob": q,
        "next_generation_matrix": next_generation_matrix(q, CONTACTS, POPULATION, GAMMA),
        "ifr_by_group": dict(zip(GROUPS, IFR)),
        "with_100k_doses": {s: {"infections": float(r["infections"]), "deaths": float(r["deaths"])}
                            for s, r in at.iterrows()},
        "best_for_infections_100k": str(at["infections"].idxmin()),
        "best_for_deaths_100k": str(at["deaths"].idxmin()),
    }
    save_json("age_vaccination_summary", results)
    return results


if __name__ == "__main__":
    print(main())
