"""M4 analysis: epidemics on three network types, superspreaders, targeted vaccination."""

from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from analysis.common import save_csv, save_json
from src.network import (
    NETWORK_TYPES,
    build_network,
    choose_vaccinated,
    degree_stats,
    network_r0,
    rank_nodes,
    simulate_network_sir,
)
from src.style import CATEGORICAL, COMPARTMENT_COLORS, savefig

N_NODES, MEAN_DEGREE = 2000, 8
TAU, GAMMA = 0.06, 0.2          # per-contact transmission rate, recovery rate (per day)
N_SEEDS, DAYS = 5, 200
N_RUNS = 50
COVERAGES = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30)
VACC_RUNS = 20
STRATEGIES = ("random", "degree", "betweenness")
NET_COLORS = dict(zip(NETWORK_TYPES, CATEGORICAL))


def compare_networks(graphs: dict) -> tuple[pd.DataFrame, dict]:
    """Run N_RUNS epidemics per network; return a summary table and I(t) matrices."""
    rows, curves = [], {}
    for kind, G in graphs.items():
        runs = [simulate_network_sir(G, TAU, GAMMA, N_SEEDS, DAYS, seed=s)[0] for s in range(N_RUNS)]
        I = np.array([r.I for r in runs]) / N_NODES
        curves[kind] = I
        attack = np.array([r.attack_rate for r in runs])
        rows.append({"network": kind, **degree_stats(G),
                     "approx_R0": network_r0(G, TAU, GAMMA),
                     "attack_rate_mean": attack.mean(),
                     "attack_rate_sd": attack.std(ddof=1),
                     "peak_prevalence_mean": I.max(axis=1).mean(),
                     "peak_day_mean": I.argmax(axis=1).mean()})
    return pd.DataFrame(rows), curves


def fig_curves(curves: dict) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.6))
    days = np.arange(DAYS + 1)
    for kind, I in curves.items():
        lo, hi = np.percentile(I * 100, [5, 95], axis=0)
        ax.fill_between(days, lo, hi, color=NET_COLORS[kind], alpha=0.18)
        ax.plot(days, I.mean(axis=0) * 100, color=NET_COLORS[kind], label=kind)
    ax.set_xlim(0, 120)
    ax.set_xlabel("Day")
    ax.set_ylabel("Infectious, % of nodes")
    ax.set_title(f"Same mean degree ({MEAN_DEGREE}), different structure: mean and 90% band "
                 f"of {N_RUNS} runs", fontsize=11.5)
    ax.legend()
    savefig(fig, "m4_network_epidemic_curves")


def fig_degree_distribution(graphs: dict) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.3))
    for kind, G in graphs.items():
        k = np.array([d for _, d in G.degree()])
        values, counts = np.unique(k, return_counts=True)
        ax1.plot(values, counts / counts.sum(), "o-", ms=3, lw=1.2, color=NET_COLORS[kind],
                 label=kind)
        ks = np.sort(k)
        ccdf = 1 - np.arange(len(ks)) / len(ks)
        ax2.loglog(ks, ccdf, color=NET_COLORS[kind], label=kind)
    ax1.set_xlim(0, 40)
    ax1.set_xlabel("Degree k")
    ax1.set_ylabel("Fraction of nodes P(k)")
    ax1.set_title("Degree distribution (linear)")
    ax1.legend()
    ax2.set_xlabel("Degree k (log)")
    ax2.set_ylabel("P(degree >= k) (log)")
    ax2.set_title("Tail of the degree distribution: hubs in scale-free networks")
    savefig(fig, "m4_degree_distribution")


def fig_network_snapshots() -> None:
    """Small (300-node) versions of each network coloured by state at their epidemic peak."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    colors = np.array([COMPARTMENT_COLORS["S"], COMPARTMENT_COLORS["I"], COMPARTMENT_COLORS["R"]])
    for ax, kind in zip(axes, NETWORK_TYPES):
        G = build_network(kind, n=300, mean_degree=6, seed=3)
        run, snaps = simulate_network_sir(G, 0.12, GAMMA, n_seeds=3, days=150, seed=3,
                                          snapshot_days=tuple(range(151)))
        peak_day = int(np.argmax(run.I))
        state = snaps[peak_day]
        pos = nx.spring_layout(G, seed=1, k=0.15)
        deg = np.array([d for _, d in G.degree()])
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.12, width=0.5)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors[state], node_size=8 + 3 * deg,
                               linewidths=0.3, edgecolors="white")
        ax.set_title(f"{kind}\nday {peak_day} (peak): {int(run.I[peak_day])} infectious")
        ax.axis("off")
    handles = [Line2D([], [], marker="o", ls="", color=c, label=l, markersize=8)
               for c, l in zip(colors, ("Susceptible", "Infectious", "Recovered"))]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Contact networks coloured by infection state (node size = degree)", y=1.02)
    savefig(fig, "m4_network_graphs")


def superspreaders(G: nx.Graph) -> dict:
    """Pool secondary infections over runs on the scale-free network; relate to centrality."""
    secondary = np.zeros(N_NODES)
    for s in range(N_RUNS):
        secondary += simulate_network_sir(G, TAU, GAMMA, N_SEEDS, DAYS, seed=s)[0].secondary
    deg = np.array([G.degree(v) for v in range(N_NODES)])
    btw_dict = nx.betweenness_centrality(G, k=500, seed=0)
    btw = np.array([btw_dict[v] for v in range(N_NODES)])
    top = rank_nodes(G, "degree")[: N_NODES // 100]
    share_top1 = secondary[top].sum() / secondary.sum()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.3))
    per_run = secondary / N_RUNS
    ax1.scatter(deg, per_run, s=8, alpha=0.4, color=NET_COLORS["Barabasi-Albert"])
    ax1.set_xscale("log")
    ax1.set_xlabel("Degree (log)")
    ax1.set_ylabel("Secondary infections per run")
    ax1.set_title("Superspreaders are the hubs")
    ax2.scatter(btw, per_run, s=8, alpha=0.4, color=CATEGORICAL[3])
    ax2.set_xscale("symlog", linthresh=1e-4)
    ax2.set_xlabel("Betweenness centrality (symlog)")
    ax2.set_ylabel("Secondary infections per run")
    ax2.set_title("...and the bridges between parts of the network")
    fig.suptitle(f"Barabasi-Albert network: top 1% of nodes by degree cause "
                 f"{share_top1:.0%} of infections", y=1.02)
    savefig(fig, "m4_superspreaders")
    from scipy.stats import spearmanr
    return {"top1pct_degree_share_of_infections": float(share_top1),
            "spearman_degree_vs_secondary": float(spearmanr(deg, secondary)[0]),
            "spearman_betweenness_vs_secondary": float(spearmanr(btw, secondary)[0])}


def vaccination_experiment(graphs: dict) -> pd.DataFrame:
    rows = []
    for kind, G in graphs.items():
        rankings = {s: rank_nodes(G, s) for s in ("degree", "betweenness")}
        for strategy in STRATEGIES:
            for cov in COVERAGES:
                attack = []
                for s in range(VACC_RUNS):
                    vacc = choose_vaccinated(G, cov, strategy, seed=s, ranking=rankings.get(strategy))
                    run, _ = simulate_network_sir(G, TAU, GAMMA, N_SEEDS, DAYS, seed=s, vaccinated=vacc)
                    attack.append(run.attack_rate)
                rows.append({"network": kind, "strategy": strategy, "coverage": cov,
                             "attack_rate_mean": float(np.mean(attack)),
                             "attack_rate_sd": float(np.std(attack, ddof=1))})
    return pd.DataFrame(rows)


def fig_vaccination(table: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharey=True)
    styles = {"random": ("o-", CATEGORICAL[0]), "degree": ("s-", CATEGORICAL[1]),
              "betweenness": ("^-", CATEGORICAL[2])}
    for ax, kind in zip(axes, NETWORK_TYPES):
        for strategy, g in table[table["network"] == kind].groupby("strategy"):
            marker, color = styles[strategy]
            ax.errorbar(g["coverage"] * 100, g["attack_rate_mean"] * 100,
                        yerr=g["attack_rate_sd"] * 100, fmt=marker, color=color,
                        capsize=3, label=f"{strategy}")
        ax.set_title(kind)
        ax.set_xlabel("Vaccination coverage (% of nodes)")
    axes[0].set_ylabel("Attack rate (% of all nodes infected)")
    axes[0].legend(title="who is vaccinated")
    fig.suptitle(f"Targeted vs random vaccination (mean +/- SD of {VACC_RUNS} runs)", y=1.02)
    savefig(fig, "m4_vaccination_strategies")


def main() -> dict:
    graphs = {k: build_network(k, N_NODES, MEAN_DEGREE, seed=0) for k in NETWORK_TYPES}
    table, curves = compare_networks(graphs)
    save_csv("m4_network_comparison", table)
    fig_curves(curves)
    fig_degree_distribution(graphs)
    fig_network_snapshots()
    spreaders = superspreaders(graphs["Barabasi-Albert"])
    vacc = vaccination_experiment(graphs)
    save_csv("m4_vaccination", vacc)
    fig_vaccination(vacc)
    results = {"setup": {"nodes": N_NODES, "mean_degree": MEAN_DEGREE, "tau": TAU,
                         "gamma": GAMMA, "initial_infected": N_SEEDS, "runs": N_RUNS},
               "superspreaders_barabasi_albert": spreaders,
               "attack_rate_mean": dict(zip(table["network"], table["attack_rate_mean"]))}
    save_json("m4_summary", results)
    return results


if __name__ == "__main__":
    print(main())
