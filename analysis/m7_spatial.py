"""M7 analysis: spatial spread across regions, travel restrictions, animated map."""

from __future__ import annotations

import imageio_ffmpeg
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.colors import LogNorm

from analysis.common import save_csv, save_json
from src.metapopulation import default_geography, gravity_flows, simulate_metapop
from src.style import CATEGORICAL, FIGURES_DIR, savefig

matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

TRAVEL_FRACTION = 0.002      # share of the total population travelling per day
RESTRICTION = 0.9            # travel restriction scenario: flows cut by 90%
PARAMS = {"beta": 0.5, "sigma": 1 / 3, "gamma": 1 / 5}   # R0 = 2.5
T_MAX = 250
SNAPSHOT_DAYS = (30, 50, 70, 90)
NORM = LogNorm(vmin=1e-5, vmax=0.2)
CMAP = "magma_r"


def draw_map(ax, geo, F, prevalence_row, day, scatter=None):
    """Regions as circles (area ~ population, colour = prevalence), edges = travel flow."""
    if scatter is None:
        w = F / F.max()
        for i in range(geo.n):
            for j in range(i + 1, geo.n):
                if w[i, j] > 0.01:
                    ax.plot(*geo.xy[[i, j]].T, color="grey", lw=0.5 + 6 * w[i, j],
                            alpha=0.35, zorder=1)
        sizes = geo.population / geo.population.max() * 1400 + 120
        scatter = ax.scatter(*geo.xy.T, s=sizes, c=np.clip(prevalence_row, 1e-7, None),
                             cmap=CMAP, norm=NORM, edgecolors="black", linewidths=0.8, zorder=2)
        for name, (x, y), size in zip(geo.names, geo.xy, sizes):
            radius_pt = np.sqrt(size) / 2  # scatter size is (diameter in points)^2
            ax.annotate(name, (x, y), xytext=(0, -radius_pt - 3), textcoords="offset points",
                        ha="center", va="top", fontsize=8.5)
        ax.set_xlim(-7, 7.3)
        ax.set_ylim(-5.8, 7.2)
        ax.set_aspect("equal")
        ax.axis("off")
    else:
        scatter.set_array(np.clip(prevalence_row, 1e-7, None))
    ax.set_title(f"Day {day:.0f}")
    return scatter


def fig_regional_curves(base, restricted) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), sharey=True)
    for ax, res, title in ((axes[0], base, "Baseline travel"),
                           (axes[1], restricted, f"Travel cut by {RESTRICTION:.0%}")):
        for k, name in enumerate(res.geo.names):
            ax.plot(res.t, res.prevalence()[:, k] * 100, color=CATEGORICAL[k % 8], label=name,
                    lw=2.4 if k == 0 else 1.6)
        ax.set_title(title)
        ax.set_xlabel("Day")
    axes[0].set_ylabel("Infectious, % of region")
    axes[0].legend(ncol=2, fontsize=8.5)
    fig.suptitle("Metapopulation SEIR: the wave starts in the Capital and travels outward", y=1.02)
    savefig(fig, "m7_regional_curves")


def fig_arrival(geo, base_table, restr_table) -> None:
    dist = geo.distances()[0]
    order = np.argsort(dist)
    y = np.arange(geo.n)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(y - 0.2, base_table["arrival_day"].to_numpy()[order], height=0.4,
            color=CATEGORICAL[0], label="baseline travel")
    ax.barh(y + 0.2, restr_table["arrival_day"].to_numpy()[order], height=0.4,
            color=CATEGORICAL[1], label=f"travel cut by {RESTRICTION:.0%}")
    ax.set_yticks(y, [f"{geo.names[k]} ({dist[k]:.1f})" for k in order])
    ax.invert_yaxis()
    ax.set_xlabel("Arrival day (prevalence first above 1 in 10,000)")
    ax.set_ylabel("Region (distance from Capital)")
    ax.set_title("Travel restrictions delay arrival, they do not prevent it")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper right")
    savefig(fig, "m7_arrival_times")


def fig_snapshots(res, F) -> None:
    fig, axes = plt.subplots(1, len(SNAPSHOT_DAYS), figsize=(18, 4.3))
    for ax, day in zip(axes, SNAPSHOT_DAYS):
        sc = draw_map(ax, res.geo, F, res.prevalence()[day], day)
    cb = fig.colorbar(sc, ax=axes, shrink=0.7, pad=0.01)
    cb.set_label("Infectious fraction (log scale)")
    fig.suptitle("Spatial spread of the epidemic (circle area = population, lines = travel)",
                 y=0.93)
    savefig(fig, "m7_spatial_snapshots")


def animate(res, F, step: int = 2, last_day: int = 180) -> dict:
    """GIF and MP4 of the wave moving across the map, with a synchronised curve panel."""
    fig = plt.figure(figsize=(11, 5), dpi=100)
    ax_map = fig.add_axes([0.0, 0.05, 0.5, 0.85])
    ax_ts = fig.add_axes([0.58, 0.15, 0.4, 0.7])
    prev = res.prevalence()
    scatter = draw_map(ax_map, res.geo, F, prev[0], 0)
    fig.colorbar(scatter, ax=ax_map, shrink=0.7, label="Infectious fraction (log)")
    for k, name in enumerate(res.geo.names):
        ax_ts.plot(res.t, prev[:, k] * 100, color=CATEGORICAL[k % 8], lw=1.5, label=name)
    marker = ax_ts.axvline(0, color="black", lw=1)
    ax_ts.set_xlabel("Day")
    ax_ts.set_ylabel("Infectious, % of region")
    ax_ts.legend(fontsize=7, ncol=2)
    ax_ts.set_title("Prevalence by region")
    ax_ts.set_xlim(0, last_day)
    frames = list(range(0, last_day + 1, step))

    def update(i):
        draw_map(ax_map, res.geo, F, prev[i], res.t[i], scatter=scatter)
        marker.set_xdata([res.t[i], res.t[i]])
        return scatter, marker

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    gif = FIGURES_DIR / "m7_spatial_spread.gif"
    mp4 = FIGURES_DIR / "m7_spatial_spread.mp4"
    anim.save(gif, writer=PillowWriter(fps=12), dpi=65)  # low dpi keeps the GIF small
    out = {"gif": gif.name, "frames": len(frames)}
    try:
        anim.save(mp4, writer=FFMpegWriter(fps=12, bitrate=1800), dpi=120)
        out["mp4"] = mp4.name
    except Exception as exc:  # MP4 is optional; the GIF always exists
        out["mp4_error"] = str(exc)
    plt.close(fig)
    return out


def main() -> dict:
    geo = default_geography()
    F = gravity_flows(geo, TRAVEL_FRACTION)
    base = simulate_metapop(geo, F, t_max=T_MAX, **PARAMS)
    restricted = simulate_metapop(geo, F * (1 - RESTRICTION), t_max=T_MAX, **PARAMS)
    base_table, restr_table = base.summary(), restricted.summary()
    base_table["distance_from_capital"] = geo.distances()[0]
    restr_table["distance_from_capital"] = geo.distances()[0]
    save_csv("m7_regions_baseline", base_table)
    save_csv("m7_regions_travel_restricted", restr_table)
    save_csv("m7_travel_matrix",
             pd.DataFrame(F, index=geo.names, columns=geo.names).rename_axis("from").reset_index())
    fig_regional_curves(base, restricted)
    fig_arrival(geo, base_table, restr_table)
    fig_snapshots(base, F)
    anim = animate(base, F)
    delay = (restr_table["arrival_day"] - base_table["arrival_day"]).iloc[1:]
    results = {
        "setup": {"regions": geo.n, "total_population": int(geo.population.sum()),
                  "daily_travel_fraction": TRAVEL_FRACTION, "R0_within_region":
                  PARAMS["beta"] / PARAMS["gamma"], "seed_region": geo.names[0]},
        "baseline_last_arrival_day": float(base_table["arrival_day"].max()),
        "restricted_last_arrival_day": float(restr_table["arrival_day"].max()),
        "mean_arrival_delay_days_outside_capital": float(delay.mean()),
        "baseline_overall_attack_rate": float(base.C[-1].sum() / geo.population.sum()),
        "restricted_overall_attack_rate": float(restricted.C[-1].sum() / geo.population.sum()),
        "animation": anim,
    }
    save_json("m7_summary", results)
    return results


if __name__ == "__main__":
    print(main())
