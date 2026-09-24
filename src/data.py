"""Data layer: download-or-synthesize, with everything cached in ``data/``.

Real sources (public):

* **1978 English boarding-school influenza outbreak** (BMJ 1978; distributed in
  the ``outbreaks`` R package by RECON). 763 boys, daily count confined to bed.
  A closed population with a single wave: the classic SIR teaching dataset.
* **JHU CSSE COVID-19 time series** (global confirmed cases and deaths).

If a download fails, nothing blocks: the fitting dataset falls back to a
synthetic outbreak generated from known parameters and is labelled
``SYNTHETIC`` everywhere it appears. The synthetic dataset is always produced
as well, because recovering its known parameters validates the fitting code.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.compartmental import ModelParams, simulate
from src.style import DATA_DIR

FLU_URL = (
    "https://raw.githubusercontent.com/reconhub/outbreaks/master/data/"
    "influenza_england_1978_school.RData"
)
JHU_BASE = (
    "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/"
    "csse_covid_19_data/csse_covid_19_time_series/"
)
JHU_FILES = {
    "confirmed": "time_series_covid19_confirmed_global.csv",
    "deaths": "time_series_covid19_deaths_global.csv",
}
FLU_CSV = "boarding_school_flu_1978.csv"
SYNTHETIC_CSV = "synthetic_outbreak.csv"
SYNTHETIC_TRUTH = "synthetic_outbreak_truth.json"

# Ground truth for the synthetic outbreak (similar in scale to the flu outbreak).
SYNTHETIC_N = 1000
SYNTHETIC_TRUTH_PARAMS = {"beta": 1.60, "gamma": 0.45, "I0": 2.0}
SYNTHETIC_DAYS = 20
SYNTHETIC_SEED = 42


@dataclass
class OutbreakData:
    """A single observed outbreak time series of infectious prevalence."""

    name: str
    t: np.ndarray            # days since first observation
    observed: np.ndarray     # observed infectious (prevalence) counts
    N: int                   # population size
    source: str
    is_synthetic: bool
    truth: dict | None = None  # known parameters (synthetic data only)

    @property
    def label(self) -> str:
        """Display name, with an explicit SYNTHETIC tag when relevant."""
        return f"{self.name} [SYNTHETIC]" if self.is_synthetic else self.name


def _download(url: str, timeout: float = 30.0) -> bytes:
    """Fetch ``url``; raises on any network or HTTP error."""
    import requests

    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def fetch_cached(url: str, filename: str, data_dir: Path = DATA_DIR) -> Path | None:
    """Return the cached file, downloading it once if needed. None on failure."""
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / filename
    if path.exists() and path.stat().st_size > 0:
        return path
    try:
        path.write_bytes(_download(url))
        return path
    except Exception as exc:  # any failure -> caller falls back
        warnings.warn(f"download failed for {url}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Boarding-school influenza
# ---------------------------------------------------------------------------

def load_boarding_school(data_dir: Path = DATA_DIR) -> OutbreakData | None:
    """Load the 1978 boarding-school flu data (download + cache). None if unavailable."""
    csv_path = data_dir / FLU_CSV
    if not csv_path.exists():
        raw = fetch_cached(FLU_URL, "influenza_england_1978_school.RData", data_dir)
        if raw is None:
            return None
        try:
            import rdata

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # R 'Date' class has no converter
                df = rdata.read_rda(str(raw))["influenza_england_1978_school"]
        except Exception as exc:
            warnings.warn(f"could not parse {raw}: {exc}")
            return None
        df = pd.DataFrame(df).reset_index(drop=True)
        # R stores dates as days since 1970-01-01.
        df["date"] = pd.to_datetime("1970-01-01") + pd.to_timedelta(
            df["date"].astype(float), unit="D")
        df.to_csv(csv_path, index=False)
    df = pd.read_csv(csv_path, parse_dates=["date"])
    return OutbreakData(
        name="Boarding-school influenza, England 1978",
        t=(df["date"] - df["date"].iloc[0]).dt.days.to_numpy(dtype=float),
        observed=df["in_bed"].to_numpy(dtype=float),
        N=763,
        source="BMJ 1978 via RECON 'outbreaks' R package (GitHub)",
        is_synthetic=False,
    )


# ---------------------------------------------------------------------------
# Synthetic outbreak (fallback + parameter-recovery validation)
# ---------------------------------------------------------------------------

def make_synthetic_outbreak(
    beta: float = SYNTHETIC_TRUTH_PARAMS["beta"],
    gamma: float = SYNTHETIC_TRUTH_PARAMS["gamma"],
    I0: float = SYNTHETIC_TRUTH_PARAMS["I0"],
    N: int = SYNTHETIC_N,
    days: int = SYNTHETIC_DAYS,
    seed: int = SYNTHETIC_SEED,
) -> OutbreakData:
    """SIR prevalence curve with Poisson observation noise and known parameters."""
    rng = np.random.default_rng(seed)
    t = np.arange(days + 1, dtype=float)
    df = simulate("SIR", ModelParams(beta=beta, gamma=gamma), N=N, I0=I0, t_eval=t)
    observed = rng.poisson(df["I"].to_numpy()).astype(float)
    return OutbreakData(
        name="Synthetic SIR outbreak",
        t=t,
        observed=observed,
        N=N,
        source=f"generated: SIR, beta={beta}, gamma={gamma}, I0={I0}, Poisson noise, seed={seed}",
        is_synthetic=True,
        truth={"beta": beta, "gamma": gamma, "I0": I0, "R0": beta / gamma},
    )


def save_synthetic(data: OutbreakData, data_dir: Path = DATA_DIR) -> Path:
    """Write the synthetic dataset and its ground truth to ``data/``."""
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / SYNTHETIC_CSV
    pd.DataFrame({"day": data.t, "infectious_observed": data.observed,
                  "dataset": "SYNTHETIC"}).to_csv(path, index=False)
    (data_dir / SYNTHETIC_TRUTH).write_text(
        json.dumps({"label": "SYNTHETIC - generated data, not a real outbreak",
                    "N": data.N, **(data.truth or {}), "source": data.source}, indent=2),
        encoding="utf-8")
    return path


def get_fit_dataset(data_dir: Path = DATA_DIR) -> OutbreakData:
    """The real flu dataset if obtainable, otherwise the labelled synthetic one."""
    real = load_boarding_school(data_dir)
    if real is not None:
        return real
    warnings.warn("real data unavailable - falling back to SYNTHETIC outbreak data")
    synthetic = make_synthetic_outbreak()
    save_synthetic(synthetic, data_dir)
    return synthetic


# ---------------------------------------------------------------------------
# JHU CSSE COVID-19
# ---------------------------------------------------------------------------

def load_jhu(kind: str = "confirmed", data_dir: Path = DATA_DIR) -> pd.DataFrame | None:
    """JHU global time series as a (date x country) DataFrame of cumulative counts."""
    path = fetch_cached(JHU_BASE + JHU_FILES[kind], f"jhu_{kind}_global.csv", data_dir)
    if path is None:
        return None
    raw = pd.read_csv(path)
    by_country = raw.drop(columns=["Province/State", "Lat", "Long"]).groupby(
        "Country/Region").sum()
    wide = by_country.T
    wide.index = pd.to_datetime(wide.index, format="%m/%d/%y")
    wide.index.name = "date"
    return wide


def daily_new(cumulative: pd.Series, smooth: int = 7) -> pd.Series:
    """Daily new counts from a cumulative series (negatives clipped, rolling mean)."""
    new = cumulative.diff().clip(lower=0).fillna(0)
    return new.rolling(smooth, center=True, min_periods=1).mean() if smooth > 1 else new


def jhu_summary(confirmed: pd.DataFrame, deaths: pd.DataFrame | None,
                countries: list[str]) -> pd.DataFrame:
    """One row per country: totals, peak daily cases and the date of the peak."""
    rows = []
    for c in countries:
        cases = confirmed[c]
        smoothed = daily_new(cases)
        first = cases[cases > 0].index.min()
        rows.append({
            "country": c,
            "first_case": first.date(),
            "last_date": cases.index.max().date(),
            "total_confirmed": int(cases.iloc[-1]),
            "total_deaths": int(deaths[c].iloc[-1]) if deaths is not None else None,
            "peak_daily_cases_7d_avg": round(float(smoothed.max()), 1),
            "peak_date": smoothed.idxmax().date(),
        })
    return pd.DataFrame(rows)

