"""Phase 2: data summary table and exploratory plots."""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.common import save_csv, save_json
from src.compartmental import ModelParams, simulate
from src.data import (
    daily_new,
    get_fit_dataset,
    jhu_summary,
    load_jhu,
    make_synthetic_outbreak,
    save_synthetic,
)
from src.style import CATEGORICAL, COMPARTMENT_COLORS, OKABE_ITO, savefig

JHU_COUNTRIES = ["US", "United Kingdom", "Italy", "Germany", "Korea, South", "India"]


def outbreak_rows(datasets) -> pd.DataFrame:
    """Summary rows for the outbreak (prevalence) datasets."""
    rows = []
    for d in datasets:
        rows.append({
            "dataset": d.label,
            "synthetic": d.is_synthetic,
            "population_N": d.N,
            "days_observed": len(d.t),
            "peak_observed_infectious": int(d.observed.max()),
            "peak_day": int(d.t[np.argmax(d.observed)]),
            "peak_pct_of_population": round(100 * d.observed.max() / d.N, 1),
            "source": d.source,
        })
    return pd.DataFrame(rows)


def fig_outbreaks(real, synthetic) -> None:
    """Observed flu prevalence and the synthetic dataset side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    ax = axes[0]
    ax.bar(real.t, real.observed, color=COMPARTMENT_COLORS["I"], alpha=0.85,
           label="Boys confined to bed")
    ax.set_title(real.label if not real.is_synthetic else real.label)
    ax.set_xlabel("Day of outbreak")
    ax.set_ylabel("Number of cases (prevalence)")
    ax.text(0.98, 0.95, f"N = {real.N} pupils", transform=ax.transAxes, ha="right", va="top")
    ax.legend(loc="upper left")

    ax = axes[1]
    truth = synthetic.truth
    t_fine = np.linspace(0, synthetic.t[-1], 400)
    curve = simulate("SIR", ModelParams(beta=truth["beta"], gamma=truth["gamma"]),
                     N=synthetic.N, I0=truth["I0"], t_eval=t_fine)
    ax.plot(t_fine, curve["I"], color="black", lw=1.5,
            label=f"true SIR curve (R0 = {truth['R0']:.2f})")
    ax.scatter(synthetic.t, synthetic.observed, color=OKABE_ITO["vermillion"], zorder=3,
               label="observed (Poisson noise)")
    ax.set_title("SYNTHETIC outbreak with known parameters")
    ax.set_xlabel("Day")
    ax.set_ylabel("Infectious (prevalence)")
    ax.legend(loc="upper right")
    savefig(fig, "data_outbreaks")


def fig_jhu(confirmed: pd.DataFrame) -> None:
    """Daily new confirmed COVID-19 cases (7-day average) for several countries."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    for ax, color, c in zip(axes.flat, CATEGORICAL, JHU_COUNTRIES):
        s = daily_new(confirmed[c])
        ax.fill_between(s.index, s.to_numpy() / 1000, color=color, alpha=0.3)
        ax.plot(s.index, s.to_numpy() / 1000, color=color, lw=1.4)
        ax.set_title(c)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for ax in axes[:, 0]:
        ax.set_ylabel("Daily new cases (thousands)")
    fig.suptitle("JHU CSSE COVID-19: daily confirmed cases, 7-day average (real data)", y=1.0)
    fig.tight_layout()
    savefig(fig, "data_jhu_daily_cases")


def main() -> dict:
    real = get_fit_dataset()
    synthetic = make_synthetic_outbreak()
    save_synthetic(synthetic)
    table = outbreak_rows([real, synthetic])
    save_csv("data_summary_outbreaks", table)
    fig_outbreaks(real, synthetic)

    out = {"fit_dataset": real.label, "fit_dataset_is_synthetic": real.is_synthetic,
           "jhu_available": False}
    confirmed = load_jhu("confirmed")
    deaths = load_jhu("deaths")
    if confirmed is not None:
        jhu_table = jhu_summary(confirmed, deaths, JHU_COUNTRIES)
        save_csv("data_summary_jhu", jhu_table)
        fig_jhu(confirmed)
        out.update(jhu_available=True, jhu_first_date=str(confirmed.index.min().date()),
                   jhu_last_date=str(confirmed.index.max().date()),
                   jhu_n_countries=int(confirmed.shape[1]))
    save_json("data_summary", out)
    return out


if __name__ == "__main__":
    print(main())
