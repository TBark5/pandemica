"""M2 stretch goal: Bayesian (MCMC) fit of SIR to the flu outbreak, compared with the bootstrap."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from analysis.common import load_json, save_json
from src.bayes import run_mcmc, sir_prevalence_fast
from src.data import get_fit_dataset, make_synthetic_outbreak
from src.fitting import bootstrap_fit
from src.style import COMPARTMENT_COLORS, OKABE_ITO, savefig

N_PREDICTIVE = 400  # posterior draws used for the predictive bands


def summarise(res) -> dict:
    out = {name: dict(zip(("median", "ci95_low", "ci95_high"), res.interval(name)))
           for name in ("R0", "beta", "gamma", "I0", "k")}
    out["sampler"] = {"walkers": res.n_walkers, "steps": res.n_steps, "burn_in": res.burn_in,
                      "kept_samples": len(res.samples),
                      "acceptance_fraction": res.acceptance_fraction,
                      "max_autocorr_time_steps": float(np.max(res.autocorr_time))}
    return out


def fig_posterior(data, res, boot_r0: np.ndarray) -> None:
    rng = np.random.default_rng(0)
    draws = res.samples[rng.choice(len(res.samples), N_PREDICTIVE, replace=False)]
    t_fine = np.linspace(data.t[0], data.t[-1], 150)
    curves = np.array([sir_prevalence_fast(t_fine, b, g, i0, data.N) for b, g, i0, _k in draws])
    # posterior predictive: add negative-binomial observation noise at the data days
    mu_obs = np.array([sir_prevalence_fast(data.t, b, g, i0, data.N) for b, g, i0, _k in draws])
    k = draws[:, 3][:, None]
    y_rep = rng.negative_binomial(k, k / (k + np.clip(mu_obs, 1e-9, None)))

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), gridspec_kw={"width_ratios": [1.5, 1, 1]})
    ax = axes[0]
    plo, phi = np.percentile(y_rep, [2.5, 97.5], axis=0)
    ax.fill_between(data.t, plo, phi, step="mid", color=OKABE_ITO["sky"], alpha=0.35,
                    label="95% posterior predictive (new data)")
    lo, hi = np.percentile(curves, [2.5, 97.5], axis=0)
    ax.fill_between(t_fine, lo, hi, color=COMPARTMENT_COLORS["I"], alpha=0.35,
                    label="95% credible band, mean curve")
    ax.plot(t_fine, np.median(curves, axis=0), color=COMPARTMENT_COLORS["I"], label="posterior median")
    ax.scatter(data.t, data.observed, color="black", zorder=3, s=26, label="observed")
    ax.set_xlabel("Day")
    ax.set_ylabel("Number infectious")
    ax.set_title("Bayesian SIR fit (negative-binomial likelihood)")
    ax.legend(fontsize=8.5)

    ax = axes[1]
    both = np.concatenate([res.r0, boot_r0])
    bins = np.linspace(*np.percentile(both, [0.1, 99.9]), 45)  # ignore extreme tails
    ax.hist(res.r0, bins=bins, density=True, alpha=0.6, color=OKABE_ITO["blue"], label="MCMC posterior")
    ax.hist(boot_r0, bins=bins, density=True, alpha=0.5, color=OKABE_ITO["orange"],
            label="least-squares bootstrap")
    ax.set_xlabel("R0")
    ax.set_ylabel("Density")
    ax.set_title("R0: posterior vs bootstrap")
    ax.legend(fontsize=8.5)

    ax = axes[2]
    ax.scatter(res.samples[::3, 0], res.samples[::3, 1], s=3, alpha=0.3, color=OKABE_ITO["purple"])
    ax.set_xlabel("beta (per day)")
    ax.set_ylabel("gamma (per day)")
    ax.set_title("Joint posterior of beta and gamma")
    savefig(fig, "m2_mcmc_posterior")


def main() -> dict:
    data = get_fit_dataset()
    res = run_mcmc(data.t, data.observed, data.N, seed=3)
    # reuse the saved bootstrap setting (500 refits, seed 1) for the comparison histogram
    boot = bootstrap_fit(data.t, data.observed, data.N, n_boot=500, seed=1)
    fig_posterior(data, res, boot.boot[:, 0] / boot.boot[:, 1])

    syn = make_synthetic_outbreak()
    syn_res = run_mcmc(syn.t, syn.observed, syn.N, seed=4)
    med, lo, hi = syn_res.interval("R0")
    results = {
        "dataset": data.label,
        "posterior": summarise(res),
        "bootstrap_R0_ci95_for_comparison": load_json("m2_summary")["fit"]["R0_ci95"],
        "synthetic_recovery": {"true_R0": syn.truth["R0"], "R0_median": med,
                               "R0_ci95": [lo, hi], "true_inside_ci": bool(lo <= syn.truth["R0"] <= hi)},
    }
    save_json("m2_bayes_summary", results)
    return results


if __name__ == "__main__":
    print(main())
