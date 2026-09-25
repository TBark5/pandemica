"""Dashboard tabs for M4 (network), M5 (interventions), M6 (sensitivity), M7 (spatial)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from dashboard import ui
from dashboard.common import load_result, show, show_figure
from src.interventions import Intervention, outcomes, run_scenario
from src.metapopulation import default_geography, gravity_flows, simulate_metapop
from src.network import (
    NETWORK_TYPES,
    build_network,
    choose_vaccinated,
    degree_stats,
    network_r0,
    rank_nodes,
    simulate_network_sir,
)
from src.sensitivity import DEFAULT_RANGES, latin_hypercube, prcc, run_samples
from src.style import CATEGORICAL, COMPARTMENT_COLORS


@st.cache_data(show_spinner="Simulating network epidemics...")
def _network_runs(kind, n, k, tau, gamma, strategy, coverage, runs):
    G = build_network(kind, n, k, seed=0)
    ranking = rank_nodes(G, strategy) if coverage and strategy != "random" else None
    out = []
    for s in range(runs):
        vacc = choose_vaccinated(G, coverage, strategy, seed=s, ranking=ranking) if coverage else None
        run, _ = simulate_network_sir(G, tau, gamma, n_seeds=5, days=200, seed=s, vaccinated=vacc)
        out.append((run.I, run.attack_rate))
    degrees = np.array([d for _, d in G.degree()])
    return np.array([o[0] for o in out]), np.array([o[1] for o in out]), degrees, \
        degree_stats(G), network_r0(G, tau, gamma)


def m4_tab() -> None:
    ui.module_header("m4", "contact networks")
    left, right = st.columns([1, 3])
    with ui.controls(left, "m4"):
        kind = st.selectbox("Network", NETWORK_TYPES, index=2)
        n = st.select_slider("Nodes", [500, 1000, 2000], value=1000)
        k = st.slider("Mean degree", 4, 16, 8, 2)
        tau = st.slider("Per-contact transmission rate tau", 0.01, 0.3, 0.06, 0.01)
        gamma = st.slider("Recovery rate gamma", 0.05, 0.5, 0.2, 0.05)
        strategy = st.radio("Vaccinate", ["random", "degree", "betweenness"], horizontal=True)
        coverage = st.slider("Vaccination coverage", 0.0, 0.4, 0.0, 0.05)
    I, attack, degrees, stats, r0 = _network_runs(kind, n, k, tau, gamma, strategy, coverage, 20)
    with right:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        for row in I:
            ax1.plot(row / n * 100, color=COMPARTMENT_COLORS["I"], alpha=0.25, lw=0.8)
        ax1.plot(I.mean(axis=0) / n * 100, color="black", lw=2, label="mean of 20 runs")
        ax1.set_xlim(0, 150)
        ax1.set_xlabel("Day")
        ax1.set_ylabel("Infectious, % of nodes")
        ax1.legend()
        ax2.hist(degrees, bins=np.arange(degrees.max() + 2) - 0.5, color=CATEGORICAL[0])
        ax2.set_yscale("log")
        ax2.set_xlabel("Degree")
        ax2.set_ylabel("Nodes (log)")
        ax2.set_title("Degree distribution")
        show(fig)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean attack rate", f"{attack.mean():.1%}")
        c2.metric("Approx. network R0", f"{r0:.2f}")
        c3.metric("Max degree", stats["max_degree"])
        c4.metric("Clustering", f"{stats['clustering']:.3f}")
    show_figure("m4_vaccination_strategies.png", "Saved run: targeted vs random vaccination")


def m5_tab() -> None:
    ui.module_header("m5", "SEIRD, R0 = 2.5")
    left, right = st.columns([1, 3])
    ivs = []
    with ui.controls(left, "m5"):
        if st.checkbox("Lockdown", True):
            ivs.append(Intervention("lockdown", st.slider("Lockdown start day", 0, 200, 60),
                                    st.slider("Contact reduction", 0.0, 0.9, 0.6, 0.05),
                                    st.slider("Lockdown length (days)", 10, 180, 60, 10)))
        if st.checkbox("Vaccination"):
            ivs.append(Intervention("vaccination", st.slider("Vaccination start day", 0, 200, 60),
                                    st.slider("% of susceptibles per day", 0.0, 2.0, 0.5, 0.1) / 100))
        if st.checkbox("Testing & isolation"):
            ivs.append(Intervention("isolation", st.slider("Isolation start day", 0, 200, 30),
                                    st.slider("Isolation rate kappa (per day)", 0.0, 0.4, 0.1, 0.02)))
    base, scen = run_scenario([]), run_scenario(ivs)
    with right:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        for df, label, ls in ((base, "no intervention", "--"), (scen, "with interventions", "-")):
            ax1.plot(df["t"], df["I"] / 1000, ls=ls, label=label)
            ax2.plot(df["t"], df["D"] / 1000, ls=ls, label=label)
        ax1.set_ylabel("Infectious (thousands)")
        ax2.set_ylabel("Cumulative deaths (thousands)")
        for ax in (ax1, ax2):
            ax.set_xlabel("Day")
            ax.set_xlim(0, 500)
        ax1.legend()
        show(fig)
        o_base, o = outcomes(base), outcomes(scen)
        c1, c2, c3 = st.columns(3)
        c1.metric("Peak infectious", f"{o['peak_infectious']:,.0f}",
                  f"{o['peak_infectious'] / o_base['peak_infectious'] - 1:+.1%}", delta_color="inverse")
        c2.metric("Deaths (2 years)", f"{o['cumulative_deaths']:,.0f}",
                  f"{o['cumulative_deaths'] / o_base['cumulative_deaths'] - 1:+.1%}",
                  delta_color="inverse")
        c3.metric("Peak day", f"{o['peak_day']:.0f}", f"{o['peak_day'] - o_base['peak_day']:+.0f} days",
                  delta_color="off")
    show_figure("m5_intervention_heatmaps.png", "Saved run: start day x strength heatmaps")


@st.cache_data(show_spinner="Running Latin hypercube samples...")
def _lhs(n: int, outcome: str) -> pd.DataFrame:
    samples = latin_hypercube(DEFAULT_RANGES, n, seed=11)
    data = run_samples(samples)
    return prcc(data[list(DEFAULT_RANGES)], data[outcome])


def m6_tab() -> None:
    ui.module_header("m6", "LHS + PRCC")
    left, right = st.columns([1, 3])
    with ui.controls(left, "m6", "Re-run live"):
        n = st.select_slider("Samples", [50, 100, 200, 300], value=100)
        outcome = st.selectbox("Outcome", ["cumulative_deaths", "peak_infectious", "peak_day",
                                           "attack_rate"])
        run = st.button("Run sensitivity analysis", type="primary", width="stretch")
        st.caption("Fewer samples than the saved run, so it finishes in seconds.")
    with right:
        if run:
            table = _lhs(n, outcome)
            st.dataframe(table.style.format({"prcc": "{:+.3f}", "p_value": "{:.2g}"}),
                         hide_index=True, width="stretch")
        show_figure("m6_tornado.png", "Saved run (800 Latin hypercube samples)")
        saved = load_result("m6_summary")
        if saved:
            st.caption(f"Ranges sampled: {saved['ranges']}")


def m7_tab() -> None:
    ui.module_header("m7", "metapopulation")
    left, right = st.columns([1, 3])
    with ui.controls(left, "m7"):
        travel = st.slider("Share of population travelling per day (%)", 0.01, 1.0, 0.2, 0.01) / 100
        cut = st.slider("Travel restriction (% of flows removed)", 0, 99, 0, 1) / 100
        r0 = st.slider("R0 within regions", 1.2, 4.0, 2.5, 0.1)
    geo = default_geography()
    F = gravity_flows(geo, travel) * (1 - cut)
    res = simulate_metapop(geo, F, beta=r0 / 5, sigma=1 / 3, gamma=1 / 5, t_max=300)
    with right:
        fig, ax = plt.subplots(figsize=(10, 4))
        for kx, name in enumerate(geo.names):
            ax.plot(res.t, res.prevalence()[:, kx] * 100, color=CATEGORICAL[kx % 8], label=name)
        ax.set_xlabel("Day")
        ax.set_ylabel("Infectious, % of region")
        ax.legend(ncol=2, fontsize=8)
        show(fig)
        st.dataframe(res.summary().round(3), hide_index=True)
    show_figure("m7_spatial_spread.gif", "Saved animation of the baseline scenario")
