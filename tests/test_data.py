"""Tests for the data layer: caching, parsing and the synthetic fallback."""

import numpy as np
import pytest

from src import data
from src.style import DATA_DIR


def test_synthetic_is_reproducible_and_labelled():
    a, b = data.make_synthetic_outbreak(), data.make_synthetic_outbreak()
    np.testing.assert_array_equal(a.observed, b.observed)
    assert a.is_synthetic and "SYNTHETIC" in a.label
    assert a.truth["R0"] == pytest.approx(a.truth["beta"] / a.truth["gamma"])


def test_failed_download_falls_back_to_synthetic(tmp_path, monkeypatch):
    """With no cache and no network the project must continue on synthetic data."""
    def fail(*_args, **_kwargs):
        raise ConnectionError("offline")

    monkeypatch.setattr(data, "_download", fail)
    with pytest.warns(UserWarning):
        ds = data.get_fit_dataset(tmp_path)
    assert ds.is_synthetic
    assert (tmp_path / data.SYNTHETIC_CSV).exists()
    assert "SYNTHETIC" in (tmp_path / data.SYNTHETIC_TRUTH).read_text()


def test_failed_jhu_download_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "_download", lambda *a, **k: (_ for _ in ()).throw(OSError()))
    with pytest.warns(UserWarning):
        assert data.load_jhu("confirmed", tmp_path) is None


@pytest.mark.skipif(not (DATA_DIR / data.FLU_CSV).exists(), reason="flu data not cached")
def test_boarding_school_data_matches_published_values():
    """The 1978 outbreak: 14 days, N = 763, peak of 298 boys in bed on day 5."""
    ds = data.load_boarding_school()
    assert len(ds.t) == 14 and ds.N == 763 and not ds.is_synthetic
    assert ds.observed.max() == 298 and ds.t[np.argmax(ds.observed)] == 5


def test_daily_new_clips_negative_corrections():
    import pandas as pd

    s = pd.Series([0, 5, 12, 10, 20])  # a downward data correction on day 3
    new = data.daily_new(s, smooth=1)
    assert (new >= 0).all() and new.iloc[-1] == 10
