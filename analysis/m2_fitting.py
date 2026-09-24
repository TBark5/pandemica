"""M2 analysis: fit SIR to the flu outbreak, recover synthetic parameters, COVID growth R0."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.common import save_csv, save_json
from src.data import daily_new, get_fit_dataset, load_jhu, make_synthetic_outbreak
from src.fitting import bootstrap_curves, bootstrap_fit, fit_growth_rate
from src.style import CATEGORICAL, COMPARTMENT_COLORS, OKABE_ITO, savefig

N_BOOT = 500
COVERAGE_DATASETS = 20  # synthetic datasets in the CI coverage check
COVERAGE_BOOT = 50      # bootstrap refits per dataset (kept small for runtime)
# Assumed natural history for the COVID-19 growth-rate conversion (see DECISIONS.md).
COVID_LATENT_DAYS = 5.2
COVID_INFECTIOUS_DAYS = 5.0
GROWTH_WINDOW_DAYS = 14
GROWTH_START_CASES = 20  # window starts when 7-day-average daily cases first reach this
COUNTRIES = ["US", "United Kingdom", "Italy", "Germany", "Korea, South", "India"]


def fig_fit(data, fit, name: str) -> None:
    """Observed vs fitted prevalence with a 95% bootstrap band, plus R0 histogram."""
    t_fine = np.linspace(data.t[0], data.t[-1], 200)
    curves = bootstrap_curves(fit, t_fine)
    lo, hi = np.percentile(curves, [2.5, 97.5], axis=0)
    central = np.median(curves, axis=0)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.8, 1]})
    ax.fill_between(t_fine, lo, hi, color=COMPARTMENT_COLORS["I"], alpha=0.25,
                    label="95% CI of fitted curve (bootstrap)")
    ax.plot(t_fine, central, color=COMPARTMENT_COLORS["I"], label="fitted SIR I(t)")
    ax.scatter(data.t, data.observed, color="black", zorder=3, s=28, label="observed")
    lo_r0, hi_r0 = fit.ci("R0")
    ax.set_title(f"{data.label}: fitted vs observed")
    ax.set_xlabel("Day")
    ax.set_ylabel("Number infectious (prevalence)")
    ax.text(0.98, 0.95, f"R0 = {fit.r0:.2f}  (95% CI {lo_r0:.2f} to {hi_r0:.2f})\n"
            f"beta = {fit.beta:.2f}/day, 1/gamma = {fit.infectious_period:.2f} days",
            transform=ax.transAxes, ha="right", va="top", fontsize=10)
    ax.legend(loc="center right")

    r0_samples = fit.boot[:, 0] / fit.boot[:, 1]
    ax2.hist(r0_samples, bins=30, color=OKABE_ITO["blue"], alpha=0.8)
    ax2.axvline(fit.r0, color="black", lw=2, label="point estimate")
    for v in (lo_r0, hi_r0):
        ax2.axvline(v, color="black", ls="--", lw=1)
    if data.truth:
        ax2.axvline(data.truth["R0"], color=OKABE_ITO["vermillion"], lw=2, label="true R0")
    ax2.set_title(f"Bootstrap distribution of R0 (n = {len(fit.boot)})")
    ax2.set_xlabel("R0")
    ax2.set_ylabel("Count")
    ax2.legend()
    savefig(fig, name)


def recovery_table(data, fit) -> pd.DataFrame:
    """True vs estimated parameters for the synthetic dataset."""
    rows = []
    for p in ("beta", "gamma", "I0", "R0"):
        est = fit.r0 if p == "R0" else getattr(fit, p)
        lo, hi = fit.ci(p)
        true = data.truth[p]
        rows.append({"parameter": p, "true": true, "estimate": est, "ci95_low": lo,
                     "ci95_high": hi, "relative_error_pct": 100 * (est - true) / true,
                     "true_inside_ci": bool(lo <= true <= hi)})
    return pd.DataFrame(rows)


def ci_coverage(scale: str = "sqrt") -> dict:
    """Refit many synthetic datasets (different noise seeds) and count how often the
    95% bootstrap CI for R0 contains the true value. Also checks for bias."""
    covered, estimates = [], []
    for seed in range(COVERAGE_DATASETS):
        d = make_synthetic_outbreak(seed=1000 + seed)
        fit = bootstrap_fit(d.t, d.observed, d.N, n_boot=COVERAGE_BOOT, seed=seed, scale=scale)
        lo, hi = fit.ci("R0")
        covered.append(lo <= d.truth["R0"] <= hi)
        estimates.append(fit.r0)
    est = np.array(estimates)
    return {"scale": scale, "n_datasets": COVERAGE_DATASETS, "n_boot_each": COVERAGE_BOOT,
            "true_R0": make_synthetic_outbreak().truth["R0"],
            "R0_ci95_coverage": float(np.mean(covered)),
            "mean_R0_estimate": float(est.mean()), "sd_R0_estimate": float(est.std(ddof=1))}


def covid_growth() -> pd.DataFrame | None:
    """Early growth-rate R0 estimates for several countries' first COVID-19 wave."""
    confirmed = load_jhu("confirmed")
    if confirmed is None:
        return None
    sigma, gamma = 1 / COVID_LATENT_DAYS, 1 / COVID_INFECTIOUS_DAYS
    rows, series = [], {}
    for c in COUNTRIES:
        smoothed = daily_new(confirmed[c])            # used only to choose the window start
        start = smoothed[smoothed >= GROWTH_START_CASES].index[0]
        raw = daily_new(confirmed[c], smooth=1).loc[start:].iloc[:GROWTH_WINDOW_DAYS]
        days = np.arange(len(raw))
        keep = raw.to_numpy() > 0                     # drop zero-report days (log undefined)
        window = pd.Series(raw.to_numpy()[keep], index=days[keep])
        g = fit_growth_rate(window.to_numpy(), sigma, gamma, days=window.index.to_numpy())
        series[c] = (window, g)
        rows.append({"country": c, "window_start": start.date(), "days": g.n_days,
                     "zero_report_days_dropped": int((~keep).sum()),
                     "growth_rate_per_day": g.r, "growth_rate_ci95_low": g.r_ci95[0],
                     "growth_rate_ci95_high": g.r_ci95[1], "doubling_time_days": g.doubling_time,
                     "R0": g.r0, "R0_ci95_low": g.r0_ci95[0], "R0_ci95_high": g.r0_ci95[1]})
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for color, (c, (window, g)) in zip(CATEGORICAL, series.items()):
        x = window.index.to_numpy(dtype=float)
        ax.scatter(x, window.to_numpy(), color=color, s=16)
        intercept = np.mean(np.log(window.to_numpy()) - g.r * x)
        ax.plot(x, np.exp(intercept + g.r * x), color=color,
                label=f"{c}: r = {g.r:.2f}/day, R0 = {g.r0:.1f}")
    ax.set_yscale("log")
    ax.set_xlabel(f"Days since 7-day-average cases reached {GROWTH_START_CASES}/day")
    ax.set_ylabel("Daily new confirmed cases, raw (log scale)")
    ax.set_title("Early COVID-19 growth (JHU CSSE) and implied R0")
    ax.legend(fontsize=8.5)
    savefig(fig, "m2_covid_growth")
    return pd.DataFrame(rows)


def main() -> dict:
    real = get_fit_dataset()
    real_fit = bootstrap_fit(real.t, real.observed, real.N, n_boot=N_BOOT, seed=1)
    fig_fit(real, real_fit, "m2_fit_observed")

    synthetic = make_synthetic_outbreak()
    syn_fit = bootstrap_fit(synthetic.t, synthetic.observed, synthetic.N, n_boot=N_BOOT, seed=2)
    fig_fit(synthetic, syn_fit, "m2_fit_synthetic_recovery")
    table = recovery_table(synthetic, syn_fit)
    save_csv("m2_parameter_recovery", table)

    results = {
        "dataset": real.label,
        "dataset_is_synthetic": real.is_synthetic,
        "fit": real_fit.summary(),
        "synthetic_recovery": {"truth": synthetic.truth, "fit": syn_fit.summary(),
                               "all_true_values_inside_ci": bool(table["true_inside_ci"].all())},
    }
    results["ci_coverage"] = ci_coverage("sqrt")
    # Same check with least squares on raw counts: shows why the sqrt scale is used.
    results["ci_coverage_raw_counts"] = ci_coverage("raw")
    growth = covid_growth()
    if growth is not None:
        save_csv("m2_covid_growth_r0", growth)
        results["covid_assumptions"] = {"latent_days": COVID_LATENT_DAYS,
                                        "infectious_days": COVID_INFECTIOUS_DAYS,
                                        "window_days": GROWTH_WINDOW_DAYS,
                                        "start_threshold_daily_cases": GROWTH_START_CASES}
    save_json("m2_summary", results)
    return results


if __name__ == "__main__":
    print(main())
