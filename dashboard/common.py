"""Helpers shared by the dashboard tabs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

from src.style import FIGURES_DIR, RESULTS_DIR  # noqa: F401  (also applies the plot style)


def load_result(name: str) -> dict | None:
    """Saved results/<name>.json, or None if run_all.py has not been run yet."""
    path = RESULTS_DIR / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def show_figure(name: str, caption: str | None = None) -> None:
    """Display a saved figure from figures/ if it exists."""
    path: Path = FIGURES_DIR / name
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.info(f"{name} not found - run `python run_all.py` first.")


def show(fig: plt.Figure) -> None:
    """Render a matplotlib figure in Streamlit and free its memory."""
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)
