"""Tests for M6: Latin hypercube sampling and PRCC."""

import numpy as np
import pandas as pd
import pytest

from src.sensitivity import DEFAULT_RANGES, latin_hypercube, prcc, run_samples


def test_lhs_hits_every_slice_exactly_once():
    n = 50
    s = latin_hypercube({"a": (0, 1), "b": (10, 20)}, n, seed=0)
    for col, (lo, hi) in (("a", (0, 1)), ("b", (10, 20))):
        slices = np.floor((s[col] - lo) / (hi - lo) * n).astype(int)
        assert sorted(slices) == list(range(n))


def test_prcc_detects_monotone_effects_and_ignores_noise():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.random((500, 3)), columns=["up", "down", "irrelevant"])
    y = X["up"] ** 2 - X["down"] ** 3 + 0.02 * rng.standard_normal(500)
    table = prcc(X, y).set_index("parameter")
    assert table.loc["up", "prcc"] > 0.9
    assert table.loc["down", "prcc"] < -0.9
    assert abs(table.loc["irrelevant", "prcc"]) < 0.15
    assert table.loc["irrelevant", "p_value"] > 0.01
    assert table.index[0] in ("up", "down")  # sorted by |PRCC|


def test_prcc_is_invariant_to_monotone_transforms():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.random((200, 2)), columns=["a", "b"])
    y = X["a"] + 0.3 * X["b"] + 0.1 * rng.random(200)
    p1 = prcc(X, y).set_index("parameter")["prcc"]
    p2 = prcc(X, np.log(y + 1) ** 3).set_index("parameter")["prcc"]
    pd.testing.assert_series_equal(p1.sort_index(), p2.sort_index())


def test_seird_sensitivity_signs_make_sense():
    """Deaths rise with beta and IFR; IFR has no effect on peak prevalence."""
    samples = latin_hypercube(DEFAULT_RANGES, 60, seed=3)
    data = run_samples(samples)
    X = data[list(DEFAULT_RANGES)]
    deaths = prcc(X, data["cumulative_deaths"]).set_index("parameter")["prcc"]
    peak = prcc(X, data["peak_infectious"]).set_index("parameter")
    assert deaths["beta"] > 0.5 and deaths["ifr"] > 0.3
    assert peak.loc["ifr", "p_value"] > 0.01
