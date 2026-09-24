"""M4 - SIR epidemics on contact networks (agent-based, discrete time).

Each node is a person with state S (0), I (1) or R (2). Every day:

* each infectious node infects each susceptible neighbour independently with
  probability ``p = 1 - exp(-tau)`` (``tau`` = per-contact transmission rate/day),
  so a susceptible node with ``k`` infectious neighbours is infected with
  probability ``1 - (1 - p)^k``;
* each infectious node recovers with probability ``1 - exp(-gamma)``.

This is vectorised with a sparse adjacency matrix, so one run on a few thousand
nodes takes milliseconds. For every new infection an infector is picked at random
among the node's infectious neighbours, which lets us count secondary infections
per node and find superspreaders.

Network types (all built with the same mean degree so they are comparable):
Erdos-Renyi (random), Watts-Strogatz (small-world, clustered),
Barabasi-Albert (scale-free, heavy-tailed degree distribution with hubs).
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import scipy.sparse as sp

S_STATE, I_STATE, R_STATE = 0, 1, 2
NETWORK_TYPES: tuple[str, ...] = ("Erdos-Renyi", "Watts-Strogatz", "Barabasi-Albert")


def build_network(kind: str, n: int = 2000, mean_degree: int = 8, seed: int = 0) -> nx.Graph:
    """Build one of the three network types with (approximately) the given mean degree."""
    if kind == "Erdos-Renyi":
        return nx.fast_gnp_random_graph(n, mean_degree / (n - 1), seed=seed)
    if kind == "Watts-Strogatz":
        return nx.connected_watts_strogatz_graph(n, mean_degree, 0.1, seed=seed)
    if kind == "Barabasi-Albert":
        return nx.barabasi_albert_graph(n, mean_degree // 2, seed=seed)
    raise ValueError(f"unknown network type {kind!r}; choose from {NETWORK_TYPES}")


def degree_stats(G: nx.Graph) -> dict:
    """Mean degree, max degree, <k^2>/<k> - 1 (mean excess degree) and clustering."""
    k = np.array([d for _, d in G.degree()], dtype=float)
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "mean_degree": float(k.mean()),
        "max_degree": int(k.max()),
        "mean_excess_degree": float((k**2).mean() / k.mean() - 1),
        "clustering": float(nx.average_clustering(G)),
    }


def network_r0(G: nx.Graph, tau: float, gamma: float) -> float:
    """Approximate R0 on a network: T * (<k^2>/<k> - 1).

    T = tau / (tau + gamma) is the probability an infection passes along one edge
    before the infector recovers (continuous-time approximation). The factor is the
    mean *excess* degree: a newly infected node was reached through one edge, so it
    has on average <k^2>/<k> - 1 other neighbours to infect. Hubs raise <k^2>.
    """
    return tau / (tau + gamma) * degree_stats(G)["mean_excess_degree"]


@dataclass
class NetworkRun:
    """Result of one network epidemic."""

    S: np.ndarray                # daily counts
    I: np.ndarray
    R: np.ndarray
    final_state: np.ndarray      # per-node state at the end
    secondary: np.ndarray        # per-node number of people they infected
    infection_day: np.ndarray    # per-node day infected (-1 if never)

    @property
    def attack_rate(self) -> float:
        """Fraction of the non-vaccinated population ever infected."""
        return float((self.infection_day >= 0).sum() / len(self.final_state))


def simulate_network_sir(
    G: nx.Graph,
    tau: float,
    gamma: float,
    n_seeds: int = 5,
    days: int = 200,
    seed: int = 0,
    vaccinated: np.ndarray | None = None,
    snapshot_days: tuple[int, ...] = (),
) -> tuple[NetworkRun, dict[int, np.ndarray]]:
    """Run one discrete-time SIR epidemic on ``G``.

    ``vaccinated`` nodes start in R (perfect vaccine). Returns the run and a dict
    of per-node state snapshots for the requested days.
    """
    rng = np.random.default_rng(seed)
    n = G.number_of_nodes()
    A = sp.csr_matrix(nx.to_scipy_sparse_array(G, nodelist=range(n), format="csr"))
    state = np.full(n, S_STATE, dtype=np.int8)
    if vaccinated is not None and len(vaccinated):
        state[vaccinated] = R_STATE
    candidates = np.flatnonzero(state == S_STATE)
    seeds = rng.choice(candidates, size=min(n_seeds, len(candidates)), replace=False)
    state[seeds] = I_STATE
    infection_day = np.full(n, -1)
    infection_day[seeds] = 0
    secondary = np.zeros(n, dtype=int)
    p_inf, p_rec = 1 - np.exp(-tau), 1 - np.exp(-gamma)
    S_t, I_t, R_t, snaps = [], [], [], {}

    for day in range(days + 1):
        S_t.append(int((state == S_STATE).sum()))
        I_t.append(int((state == I_STATE).sum()))
        R_t.append(int((state == R_STATE).sum()))
        if day in snapshot_days:
            snaps[day] = state.copy()
        infectious = state == I_STATE
        if not infectious.any():
            continue
        k_inf = A @ infectious.astype(float)  # infectious neighbours per node
        prob = 1 - (1 - p_inf) ** k_inf
        new_inf = np.flatnonzero((state == S_STATE) & (rng.random(n) < prob))
        recover = infectious & (rng.random(n) < p_rec)
        for v in new_inf:  # attribute each infection to a random infectious neighbour
            nbrs = A.indices[A.indptr[v]:A.indptr[v + 1]]
            secondary[rng.choice(nbrs[infectious[nbrs]])] += 1
        state[new_inf] = I_STATE
        state[recover] = R_STATE
        infection_day[new_inf] = day + 1

    run = NetworkRun(S=np.array(S_t), I=np.array(I_t), R=np.array(R_t), final_state=state,
                     secondary=secondary, infection_day=infection_day)
    return run, snaps


def rank_nodes(G: nx.Graph, by: str, seed: int = 0) -> np.ndarray:
    """Node ids sorted from most to least central (``degree`` or ``betweenness``).

    Betweenness uses networkx's sampled estimator (k = 500 sources) to stay fast.
    """
    if by == "degree":
        scores = dict(G.degree())
    elif by == "betweenness":
        scores = nx.betweenness_centrality(G, k=min(500, G.number_of_nodes()), seed=seed)
    else:
        raise ValueError("by must be 'degree' or 'betweenness'")
    return np.array(sorted(scores, key=lambda v: (-scores[v], v)))


def choose_vaccinated(G: nx.Graph, coverage: float, strategy: str, seed: int = 0,
                      ranking: np.ndarray | None = None) -> np.ndarray:
    """Pick ``coverage`` x N nodes to vaccinate: ``random``, ``degree`` or ``betweenness``."""
    n = G.number_of_nodes()
    count = int(round(coverage * n))
    if strategy == "random":
        return np.random.default_rng(seed).choice(n, size=count, replace=False)
    ranking = rank_nodes(G, strategy, seed) if ranking is None else ranking
    return ranking[:count]
