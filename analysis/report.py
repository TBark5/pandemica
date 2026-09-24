"""Build the results tables from results/ and inject them into README.md.

Every number in the README results section is generated here from the saved
files, so the README cannot drift from the actual runs. The section lives between
the RESULTS:START and RESULTS:END markers in README.md.
"""

from __future__ import annotations

import pandas as pd

from analysis.common import load_json
from src.style import PROJECT_ROOT, RESULTS_DIR

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
H_START, H_END = "<!-- HEADLINES:START -->", "<!-- HEADLINES:END -->"


def _csv(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / f"{name}.csv")


def _table(df: pd.DataFrame) -> str:
    """Minimal markdown table (avoids an extra dependency on `tabulate`)."""
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def section_m1() -> str:
    m1 = load_json("m1_summary")
    return (
        "### M1 - Validation against analytical results\n"
        "| Check | Result |\n|---|---|\n"
        f"| Max relative population-conservation error | {m1['max_conservation_error_rel']:.1e} |\n"
        f"| Max error vs final-size equation (attack fraction, R0 = 1.5-4) | "
        f"{m1['max_abs_final_size_error']:.1e} |\n"
        f"| Max error vs analytical SIR peak prevalence | {m1['max_abs_peak_error']:.1e} |\n"
    )


def section_m2() -> str:
    m2 = load_json("m2_summary")
    f = m2["fit"]
    rec = _csv("m2_parameter_recovery")
    rec_tbl = pd.DataFrame({
        "Parameter": rec["parameter"],
        "True": rec["true"].map("{:.3f}".format),
        "Estimate": rec["estimate"].map("{:.3f}".format),
        "95% CI": [f"{lo:.3f} - {hi:.3f}" for lo, hi in zip(rec["ci95_low"], rec["ci95_high"])],
        "Error": rec["relative_error_pct"].map("{:+.1f}%".format),
        "True inside CI": rec["true_inside_ci"].map({True: "yes", False: "no"}),
    })
    cov = m2["ci_coverage"]
    out = (
        f"### M2 - Fit to real data: {m2['dataset']}\n"
        "| Quantity | Estimate | 95% bootstrap CI |\n|---|---|---|\n"
        f"| R0 | **{f['R0']:.2f}** | {f['R0_ci95'][0]:.2f} - {f['R0_ci95'][1]:.2f} |\n"
        f"| beta (per day) | {f['beta']:.2f} | {f['beta_ci95'][0]:.2f} - {f['beta_ci95'][1]:.2f} |\n"
        f"| gamma (per day) | {f['gamma']:.3f} | {f['gamma_ci95'][0]:.3f} - {f['gamma_ci95'][1]:.3f} |\n"
        f"| Infectious period 1/gamma (days) | {f['infectious_period_days']:.2f} | "
        f"{1 / f['gamma_ci95'][1]:.2f} - {1 / f['gamma_ci95'][0]:.2f} |\n\n"
        f"({f['n_bootstrap']} bootstrap refits; RMSE {f['rmse']:.1f} cases.)\n\n"
        "**Parameter recovery on SYNTHETIC data** (known truth):\n\n"
        f"{_table(rec_tbl)}\n\n"
        f"**CI calibration:** over {cov['n_datasets']} synthetic datasets, the 95% CI for R0 "
        f"contained the true value {cov['R0_ci95_coverage']:.0%} of the time; mean estimate "
        f"{cov['mean_R0_estimate']:.3f} vs true {cov['true_R0']:.3f}. The same check with least "
        f"squares on raw counts gave {m2['ci_coverage_raw_counts']['R0_ci95_coverage']:.0%} "
        f"coverage, which is why the square-root scale is used (a small study: "
        f"{cov['n_datasets']} datasets, {cov['n_boot_each']} refits each).\n"
    )
    try:
        g = _csv("m2_covid_growth_r0")
        a = m2["covid_assumptions"]
        g_tbl = pd.DataFrame({
            "Country": g["country"], "Window start": g["window_start"],
            "Growth rate r (/day)": g["growth_rate_per_day"].map("{:.3f}".format),
            "Doubling time (days)": g["doubling_time_days"].map("{:.1f}".format),
            "Implied R0 (95% CI)": [f"{r:.1f} ({lo:.1f} - {hi:.1f})" for r, lo, hi in
                                    zip(g["R0"], g["R0_ci95_low"], g["R0_ci95_high"])],
        })
        out += (f"\n**COVID-19 early growth (JHU CSSE)**: method demonstration only; assumes a "
                f"{a['latent_days']}-day latent and {a['infectious_days']}-day infectious period.\n\n"
                f"{_table(g_tbl)}\n")
    except FileNotFoundError:
        pass
    return out


def section_m3() -> str:
    m3 = load_json("m3_summary")
    s = m3["setup"]
    return (
        f"### M3 - Gillespie stochastic SIR (N = {s['N']}, I0 = {s['I0']}, R0 = {s['R0']:.1f}, "
        f"{s['n_reps']} replicates)\n"
        "| Quantity | Simulated | Theory / ODE |\n|---|---|---|\n"
        f"| Early extinction probability | {m3['extinction_probability_simulated']:.3f} | "
        f"{m3['extinction_probability_theory']:.3f} ((1/R0)^I0) |\n"
        f"| Attack rate of major outbreaks | {m3['major_outbreak_mean_attack_rate']:.3f} | "
        f"{m3['final_size_equation_attack_rate']:.3f} (final-size equation) |\n"
        f"| Peak infectious, major outbreaks: median (5-95%) | "
        f"{m3['major_outbreak_peak_I_median']:.0f} ({m3['major_outbreak_peak_I_p5_p95'][0]:.0f} - "
        f"{m3['major_outbreak_peak_I_p5_p95'][1]:.0f}) | {m3['deterministic_peak_I']:.0f} (ODE) |\n\n"
        f"Across {len(_csv('m3_extinction'))} (R0, I0) combinations the largest gap between simulated and theoretical "
        f"extinction probability was {m3['extinction_max_abs_error']:.3f}. The gap between the "
        f"stochastic mean and the ODE shrinks with population size with log-log slope "
        f"{m3['convergence_loglog_slope']:.2f}.\n"
    )


def section_m4() -> str:
    m4 = load_json("m4_summary")
    net = _csv("m4_network_comparison")
    tbl = pd.DataFrame({
        "Network": net["network"], "Max degree": net["max_degree"],
        "Clustering": net["clustering"].map("{:.3f}".format),
        "Approx. R0": net["approx_R0"].map("{:.2f}".format),
        "Attack rate": net["attack_rate_mean"].map("{:.1%}".format),
        "Peak prevalence": net["peak_prevalence_mean"].map("{:.1%}".format),
        "Peak day": net["peak_day_mean"].map("{:.0f}".format),
    })
    v = _csv("m4_vaccination")
    v10 = v[v["coverage"].round(2) == 0.10].pivot(index="network", columns="strategy",
                                                   values="attack_rate_mean")
    v_tbl = pd.DataFrame({"Network": v10.index,
                          **{f"{c} (10% vaccinated)": v10[c].map("{:.1%}".format).to_numpy()
                             for c in ("random", "degree", "betweenness")}})
    ss = m4["superspreaders_barabasi_albert"]
    return (
        f"### M4 - Network epidemics ({m4['setup']['nodes']} nodes, mean degree "
        f"{m4['setup']['mean_degree']}, {m4['setup']['runs']} runs each)\n"
        f"{_table(tbl)}\n\nAttack rate with 10% of nodes vaccinated:\n\n{_table(v_tbl)}\n\n"
        f"On the Barabasi-Albert network the top 1% of nodes by degree caused "
        f"{ss['top1pct_degree_share_of_infections']:.0%} of all infections (Spearman correlation of "
        f"secondary infections with degree {ss['spearman_degree_vs_secondary']:.2f}, with "
        f"betweenness {ss['spearman_betweenness_vs_secondary']:.2f}).\n"
    )


def section_m5() -> str:
    m5 = load_json("m5_summary")
    c = _csv("m5_counterfactuals")
    tbl = pd.DataFrame({
        "Scenario": c["scenario"],
        "Peak infectious": c["peak_infectious"].map("{:,.0f}".format),
        "Peak day": c["peak_day"].map("{:.0f}".format),
        "Deaths": c["cumulative_deaths"].map("{:,.0f}".format),
        "Deaths averted": c["deaths_averted_pct"].map("{:.1f}%".format),
    })
    return (
        "### M5 - Interventions (SEIRD, R0 = 2.5, N = 1,000,000, 2-year horizon)\n"
        f"{_table(tbl)}\n\n"
        f"Best single 60-day lockdown in the sweep (fewest deaths): start day "
        f"{m5['lockdown_best_start_day_for_deaths']:.0f}, contact reduction "
        f"{m5['lockdown_best_strength_for_deaths']:.0%}, {m5['lockdown_best_deaths']:,.0f} deaths. "
        f"A temporary lockdown mainly delays the wave; it helps most when timed near the peak.\n"
    )


def section_m6() -> str:
    m6 = load_json("m6_summary")
    rows = []
    for outcome in ("cumulative_deaths", "peak_infectious"):
        t = _csv(f"m6_prcc_{outcome}").head(4)
        for _, r in t.iterrows():
            rows.append({"Outcome": outcome.replace("_", " "), "Parameter": r["parameter"],
                         "PRCC": f"{r['prcc']:+.2f}"})
    return (f"### M6 - Sensitivity ({m6['n_samples']} Latin hypercube samples): top 4 drivers\n"
            f"{_table(pd.DataFrame(rows))}\n")


def section_m7() -> str:
    m7 = load_json("m7_summary")
    b, r = _csv("m7_regions_baseline"), _csv("m7_regions_travel_restricted")
    tbl = pd.DataFrame({"Region": b["region"], "Population": b["population"].map("{:,}".format),
                        "Arrival day (baseline)": b["arrival_day"].map("{:.0f}".format),
                        "Arrival day (travel -90%)": r["arrival_day"].map("{:.0f}".format),
                        "Attack rate": b["attack_rate"].map("{:.1%}".format)})
    return (
        f"### M7 - Spatial spread ({m7['setup']['regions']} regions, "
        f"{m7['setup']['total_population']:,} people)\n{_table(tbl)}\n\n"
        f"Cutting travel by 90% delayed arrival outside the Capital by "
        f"{m7['mean_arrival_delay_days_outside_capital']:.1f} days on average, but the overall attack "
        f"rate stayed at {m7['restricted_overall_attack_rate']:.1%} "
        f"(vs {m7['baseline_overall_attack_rate']:.1%}).\n"
    )


def headlines() -> str:
    """Five one-line headline results for the top of the README."""
    m2, m3, m5, m7 = (load_json(n) for n in ("m2_summary", "m3_summary", "m5_summary",
                                               "m7_summary"))
    f, cov = m2["fit"], m2["ci_coverage"]
    v = _csv("m4_vaccination")
    ba = v[(v["network"] == "Barabasi-Albert") & (v["coverage"].round(2) == 0.10)]
    ba = ba.set_index("strategy")["attack_rate_mean"]
    combo = m5["counterfactuals"]["All three combined"]["deaths_averted_pct"]
    lock = m5["counterfactuals"]["Lockdown day 60, 60% for 60 d"]["deaths_averted_pct"]
    return "\n".join([
        f"- **Real-data fit:** 1978 boarding-school influenza outbreak, R0 = {f['R0']:.2f} "
        f"(95% bootstrap CI {f['R0_ci95'][0]:.2f} - {f['R0_ci95'][1]:.2f}), infectious period "
        f"{f['infectious_period_days']:.1f} days.",
        f"- **Calibrated uncertainty:** on synthetic data with known truth, the 95% CI for R0 "
        f"covered the true value in {cov['R0_ci95_coverage']:.0%} of {cov['n_datasets']} datasets.",
        f"- **Stochastic vs theory:** simulated early-extinction probability "
        f"{m3['extinction_probability_simulated']:.2f} vs branching-process theory "
        f"{m3['extinction_probability_theory']:.2f}.",
        f"- **Network structure:** vaccinating the 10% best-connected nodes of a scale-free network "
        f"cut the attack rate to {ba['degree']:.1%}, vs {ba['random']:.1%} for random vaccination.",
        f"- **Interventions:** a 60-day lockdown alone averted {lock:.1f}% of deaths (it mostly "
        f"delays the wave); combined with vaccination and isolation, {combo:.1f}%. Cutting travel "
        f"by 90% delayed regional arrival by "
        f"{m7['mean_arrival_delay_days_outside_capital']:.1f} days on average.",
    ])


def _inject(content: str, start: str, end: str, text: str) -> str:
    """Replace whatever sits between the ``start`` and ``end`` markers with ``text``."""
    if start not in content or end not in content:
        return content
    before, rest = content.split(start, 1)
    _, after = rest.split(end, 1)
    return f"{before}{start}\n{text}\n{end}{after}"


def build() -> str:
    parts = [section_m1(), section_m2(), section_m3(), section_m4(), section_m5(),
             section_m6(), section_m7()]
    return "\n".join(parts)


def main() -> dict:
    text = build()
    (RESULTS_DIR / "SUMMARY.md").write_text(
        "# Results summary (generated by analysis/report.py)\n\n## Headlines\n"
        + headlines() + "\n\n" + text, encoding="utf-8", newline="\n")
    readme = PROJECT_ROOT / "README.md"
    if readme.exists():
        content = readme.read_text(encoding="utf-8")
        content = _inject(content, START, END, text)
        content = _inject(content, H_START, H_END, headlines())
        readme.write_text(content, encoding="utf-8", newline="\n")
    return {"sections": 7}


if __name__ == "__main__":
    main()
    print((RESULTS_DIR / "SUMMARY.md").read_text(encoding="utf-8")[:3000])
