"""PANDEMICA Streamlit dashboard: one tab per module with live parameter sliders.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from dashboard import tabs_applied, tabs_core
from dashboard.common import load_result

st.set_page_config(page_title="PANDEMICA", page_icon="🦠", layout="wide")


def overview() -> None:
    st.title("PANDEMICA")
    st.markdown(
        "**Computational epidemiology and outbreak modelling platform.** Seven modules "
        "model infectious-disease spread with deterministic, stochastic, network and "
        "spatial methods, fitted to real public data.\n\n"
        "> Simplified educational models, **not** forecasting or policy tools."
    )
    m2, m3, m4 = load_result("m2_summary"), load_result("m3_summary"), load_result("m4_summary")
    if m2 and m3 and m4:
        c1, c2, c3, c4 = st.columns(4)
        lo, hi = m2["fit"]["R0_ci95"]
        c1.metric("Flu 1978 fitted R0", f"{m2['fit']['R0']:.2f}")
        c1.caption(f"95% bootstrap CI {lo:.2f} - {hi:.2f}")
        c2.metric("R0 CI coverage (synthetic)", f"{m2['ci_coverage']['R0_ci95_coverage']:.0%}")
        c3.metric("Extinction: simulated vs theory",
                  f"{m3['extinction_probability_simulated']:.2f} vs "
                  f"{m3['extinction_probability_theory']:.2f}")
        top = m4["superspreaders_barabasi_albert"]["top1pct_degree_share_of_infections"]
        c4.metric("Infections caused by top 1% hubs", f"{top:.0%}")
    else:
        st.info("Run `python run_all.py` to generate the saved results shown here.")
    st.markdown(
        "| Tab | Module | Method |\n|---|---|---|\n"
        "| M1 | Compartmental | SIR / SEIR / SEIRD ODEs (`solve_ivp`), vaccination, waning |\n"
        "| M2 | Fitting | least squares on real flu data, bootstrap CIs |\n"
        "| M3 | Stochastic | Gillespie algorithm, extinction probability |\n"
        "| M4 | Network | Erdos-Renyi / Watts-Strogatz / Barabasi-Albert, targeted vaccination |\n"
        "| M5 | Interventions | lockdown, vaccination, testing & isolation counterfactuals |\n"
        "| M6 | Sensitivity | Latin hypercube sampling + PRCC |\n"
        "| M7 | Spatial | 8-region metapopulation model with travel |"
    )


PAGES = {
    "overview": ("Overview", overview),
    "m1": ("M1 Compartmental", tabs_core.m1_tab),
    "m2": ("M2 Fitting", tabs_core.m2_tab),
    "m3": ("M3 Stochastic", tabs_core.m3_tab),
    "m4": ("M4 Network", tabs_applied.m4_tab),
    "m5": ("M5 Interventions", tabs_applied.m5_tab),
    "m6": ("M6 Sensitivity", tabs_applied.m6_tab),
    "m7": ("M7 Spatial", tabs_applied.m7_tab),
}


def main() -> None:
    # Deep link: ?tab=m5 shows a single module (handy for sharing and screenshots).
    only = st.query_params.get("tab")
    if only in PAGES:
        PAGES[only][1]()
        return
    tabs = st.tabs([label for label, _ in PAGES.values()])
    for tab, (_label, render) in zip(tabs, PAGES.values()):
        with tab:
            render()


main()
