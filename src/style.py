"""Shared plotting style for every PANDEMICA figure.

All figures use the Okabe-Ito colorblind-safe palette for categorical data
and perceptually uniform colormaps (viridis / magma / cividis) for continuous
data. Figures are saved at 300 dpi into ``figures/``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: safe for scripts, tests and CI
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"

DPI = 300

# Okabe & Ito (2008) colorblind-safe palette.
OKABE_ITO = {
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#000000",
    "grey": "#7F7F7F",
}

# Fixed color per compartment so every figure reads the same way.
COMPARTMENT_COLORS = {
    "S": OKABE_ITO["blue"],
    "E": OKABE_ITO["orange"],
    "I": OKABE_ITO["vermillion"],
    "R": OKABE_ITO["green"],
    "D": OKABE_ITO["black"],
    "V": OKABE_ITO["purple"],
}

CATEGORICAL = [
    OKABE_ITO[k]
    for k in ("blue", "vermillion", "green", "orange", "purple", "sky", "yellow", "black")
]

SEQUENTIAL_CMAP = "viridis"
HEAT_CMAP = "magma"


def apply_style() -> None:
    """Set global matplotlib rcParams for a clean, readable house style."""
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "legend.fontsize": 9.5,
            "legend.frameon": False,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "lines.linewidth": 2.0,
            "axes.prop_cycle": matplotlib.cycler(color=CATEGORICAL),
        }
    )


def savefig(fig: plt.Figure, name: str, folder: Path | None = None) -> Path:
    """Save ``fig`` as ``<folder>/<name>.png`` at 300 dpi and close it."""
    folder = folder or FIGURES_DIR
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def ensure_dirs() -> None:
    """Create the output directories if they do not exist."""
    for d in (FIGURES_DIR, RESULTS_DIR, DATA_DIR):
        d.mkdir(parents=True, exist_ok=True)


apply_style()
