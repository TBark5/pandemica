"""Dashboard tabs for M1 (compartmental), M2 (fitting) and M3 (stochastic)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from dashboard.common import load_result, show
from src.compartmental import ModelParams, final_size, simulate
from src.data import get_fit_dataset, make_synthetic_outbreak
from src.fitting import bootstrap_curves, bootstrap_fit
from src.stochastic import extinction_probability_theory, run_ensemble
from src.style import COMPARTMENT_COLORS, OKABE_ITO


def m1_tab() -> None:
    st.header("M1 - Compartmental ODE models")
    left, right = st.columns([1, 3])
    with left:
        model = st.selectbox("Model", ["SIR", "SEIR", "SEIRD"], index=2)
        r0 = st.slider("R0", 0.5, 6.0, 2.5, 0.1)
        inf_days = st.slider("Infectious period (days)", 2.0, 14.0, 7.0, 0.5)
        lat_days = st.slider("Latent period (days, SEIR/SEIRD)", 1.0, 10.0, 5.0, 0.5)
        ifr = st.slider("Infection fatality ratio (SEIRD)", 0.0, 0.05, 0.01, 0.001, format="%.3f")
        nu = st.slider("Vaccination rate (% of S per day)", 0.0, 2.0, 0.0, 0.1) / 100
        wane = st.slider("Waning immunity: mean duration (days, 0 = lifelong)", 0, 730, 0, 30)
        t_max = st.slider("Days to simulate", 60, 1000, 300, 20)
    gamma = 1 / inf_days
    params = ModelParams(beta=r0 * gamma, gamma=gamma, sigma=1 / lat_days, ifr=ifr, nu=nu,
                         omega=1 / wane if wane else 0.0)
    N = 1_000_000
    df = simulate(model, params, N=N, I0=10, t_max=t_max)
    with right:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
        for c in ("S", "E", "I", "R", "D", "V"):
            if df[c].max() > 0:
                ax1.plot(df["t"], df[c] / N * 100, color=COMPARTMENT_COLORS[c], label=c)
        ax1.set_xlabel("Day")
        ax1.set_ylabel("% of population")
        ax1.legend()
        ax1.set_title(f"{model} compartments")
        ax2.plot(df["t"], df["Reff"], color=OKABE_ITO["blue"])
        ax2.axhline(1, color="grey", ls="--")
        ax2.set_xlabel("Day")
        ax2.set_ylabel("R_eff(t)")
        ax2.set_title("Effective reproduction number")
        show(fig)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Peak infectious", f"{df['I'].max() / N:.1%}")
        c2.metric("Peak day", f"{df['t'].iloc[df['I'].idxmax()]:.0f}")
        c3.metric("Ever infected", f"{df['C'].iloc[-1] / N:.1%}")
        c4.metric("Deaths", f"{df['D'].iloc[-1]:,.0f}")
        if model in ("SIR", "SEIR") and nu == 0 and wane == 0:
            st.caption(f"Final-size equation predicts {final_size(r0, 1 - 10 / N):.2%} ever "
                       "infected (matches once the epidemic has finished).")


@st.cache_data(show_spinner="Fitting and bootstrapping...")
def _fit(use_synthetic: bool, n_boot: int):
    data = make_synthetic_outbreak() if use_synthetic else get_fit_dataset()
    fit = bootstrap_fit(data.t, data.observed, data.N, n_boot=n_boot, seed=1)
    t_fine = np.linspace(data.t[0], data.t[-1], 150)
    return data, fit, t_fine, bootstrap_curves(fit, t_fine)


def m2_tab() -> None:
    st.header("M2 - Fitting SIR to an outbreak")
    left, right = st.columns([1, 3])
    with left:
        source = st.radio("Dataset", ["Boarding-school flu 1978 (real)",
                                      "SYNTHETIC (known parameters)"])
        n_boot = st.slider("Bootstrap refits", 20, 300, 100, 20,
                           help="The saved results use 500; fewer is faster here.")
    data, fit, t_fine, curves = _fit(source.startswith("SYNTHETIC"), n_boot)
    lo, hi = np.percentile(curves, [2.5, 97.5], axis=0)
    with right:
        fig, ax = plt.subplots(figsize=(10, 4.2))
        ax.fill_between(t_fine, lo, hi, color=COMPARTMENT_COLORS["I"], alpha=0.25,
                        label="95% CI of fitted curve")
        ax.plot(t_fine, np.median(curves, axis=0), color=COMPARTMENT_COLORS["I"], label="fit")
        ax.scatter(data.t, data.observed, color="black", zorder=3, label="observed")
        ax.set_title(data.label)
        ax.set_xlabel("Day")
        ax.set_ylabel("Infectious")
        ax.legend()
        show(fig)
        r_lo, r_hi = fit.ci("R0")
        c1, c2, c3 = st.columns(3)
        c1.metric("R0", f"{fit.r0:.2f}")
        c1.caption(f"95% bootstrap CI {r_lo:.2f} - {r_hi:.2f}")
        c2.metric("beta (per day)", f"{fit.beta:.2f}")
        c3.metric("Infectious period (days)", f"{fit.infectious_period:.2f}")
        if data.truth:
            st.caption(f"True values: R0 = {data.truth['R0']:.2f}, beta = {data.truth['beta']}, "
                       f"1/gamma = {1 / data.truth['gamma']:.2f} days.")
    saved = load_result("m2_summary")
    if saved:
        cov = saved["ci_coverage"]
        st.caption(f"Saved run: 95% CI for R0 contained the true value in "
                   f"{cov['R0_ci95_coverage']:.0%} of {cov['n_datasets']} synthetic datasets.")


@st.cache_data(show_spinner="Running Gillespie replicates...")
def _ensemble(n_reps: int, N: int, I0: int, r0: float, gamma: float, t_max: int):
    return run_ensemble(n_reps, N, I0, r0 * gamma, gamma, t_max, seed=42)


def m3_tab() -> None:
    st.header("M3 - Gillespie stochastic SIR")
    left, right = st.columns([1, 3])
    with left:
        N = st.select_slider("Population N", [100, 300, 1000, 3000], value=1000)
        I0 = st.slider("Initial infected", 1, 10, 2)
        r0 = st.slider("R0 ", 0.5, 5.0, 2.5, 0.1)
        inf_days = st.slider("Infectious period (days) ", 2.0, 14.0, 10.0, 0.5)
        n_reps = st.slider("Replicates", 50, 500, 200, 50)
    t_max = 300
    ens = _ensemble(n_reps, N, I0, r0, 1 / inf_days, t_max)
    ode = simulate("SIR", ModelParams(beta=r0 / inf_days, gamma=1 / inf_days), N=N, I0=I0,
                   t_max=t_max)
    major = ens.major()
    with right:
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [2, 1]})
        for r in range(min(100, n_reps)):
            ax.plot(ens.grid, ens.I[r], lw=0.6, alpha=0.4,
                    color=COMPARTMENT_COLORS["I"] if major[r] else OKABE_ITO["grey"])
        ax.plot(ode["t"], ode["I"], color="black", ls="--", lw=2, label="ODE")
        ax.set_xlabel("Day")
        ax.set_ylabel("Infectious")
        ax.legend()
        ax.set_title("Replicates (grey = died out early)")
        ax2.hist(ens.final_size / N * 100, bins=np.linspace(0, 100, 41), color=OKABE_ITO["blue"])
        ax2.set_xlabel("Final size (% infected)")
        ax2.set_ylabel("Replicates")
        ax2.set_title("Final size distribution")
        show(fig)
        c1, c2 = st.columns(2)
        c1.metric("Extinction probability (simulated)", f"{ens.extinction_probability():.1%}")
        c2.metric("Branching theory (1/R0)^I0", f"{extinction_probability_theory(r0, I0):.1%}")
