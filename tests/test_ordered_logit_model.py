"""
Source-blind example tests for Issue #14:
feat: Elo-diff ordered logit secondary model (statsmodels)

Every test is derived from the acceptance-criteria text only.
No implementation source was read during authoring (Red-phase TDD).

Skipped criteria (per oracle report):
  - Deterministic coefficients [NOT VERIFIABLE — no concrete runtime check]
  - All tests pass [NOT VERIFIABLE — boilerplate suite gate]
  - SOLID / clean code [NOT VERIFIABLE — subjective prose]
"""

from __future__ import annotations

import importlib
from typing import cast

import numpy as np
import pandas as pd
from hypothesis import given, settings, strategies as st

from worldcup_playoff.models.dataset import MatchDataset
from worldcup_playoff.models.ordered_logit import OrderedLogitModel


# ---------------------------------------------------------------------------
# Test fixtures — built from the criteria, not from production code
# ---------------------------------------------------------------------------


def _synthetic_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """
    Minimal DataFrame matching the model's expected training schema:
      - elo_diff   : numeric feature (config-driven; sole feature per AC1)
      - y_outcome  : ordinal target — 0 = away-win, 1 = draw, 2 = home-win
                     (away < draw < home, per AC1)
    A simple sigmoid correlation makes the ordered-logit identifiable:
    higher elo_diff → more likely home win.
    """
    rng = np.random.default_rng(seed)
    elo_diff = rng.uniform(-400.0, 400.0, size=n)
    prob_home = 1.0 / (1.0 + np.exp(-elo_diff / 200.0))
    prob_away = 1.0 / (1.0 + np.exp(elo_diff / 200.0))
    prob_draw = np.full(n, 0.25)
    total = prob_home + prob_away + prob_draw
    prob_home /= total
    prob_away /= total
    prob_draw /= total
    outcomes = np.array(
        [rng.choice([0, 1, 2], p=[prob_away[i], prob_draw[i], prob_home[i]]) for i in range(n)]
    )
    return pd.DataFrame({"elo_diff": elo_diff, "y_outcome": outcomes})


class _FakeMatchDataset:
    """
    Protocol-minimal stand-in for MatchDataset.

    Satisfies the interface implied by AC4: the model's fit() must use
    .train only and must never touch .test.  Access is recorded so the
    test can assert the correct property was (or was not) read.
    """

    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        self._train = train_df
        self._test = test_df
        self.train_accessed: bool = False
        self.test_accessed: bool = False
        self.feature_cols: list[str] = ["elo_diff"]

    @property
    def train(self) -> pd.DataFrame:
        self.train_accessed = True
        return self._train

    @property
    def test(self) -> pd.DataFrame:
        self.test_accessed = True
        return self._test


def _make_dataset(n: int = 200, seed: int = 42) -> _FakeMatchDataset:
    df = _synthetic_df(n=n, seed=seed)
    split = n // 2
    return _FakeMatchDataset(
        train_df=df.iloc[:split].copy(),
        test_df=df.iloc[split:].copy(),
    )


def _fitted_model(n: int = 200, seed: int = 42) -> OrderedLogitModel:
    """Return an OrderedLogitModel already fitted on synthetic data."""
    dataset = _make_dataset(n=n, seed=seed)
    model = OrderedLogitModel()
    model.fit(cast(MatchDataset, dataset))
    return model


# ---------------------------------------------------------------------------
# AC1 — Fits an ordered logit on y_outcome (away<draw<home) using elo_diff
# ---------------------------------------------------------------------------


def test_when_fit_is_called_then_predict_succeeds():
    """AC1: fit() + predict() on elo_diff input completes without error."""
    model = _fitted_model()
    X = pd.DataFrame({"elo_diff": [0.0]})
    result = model.predict(X)
    assert result is not None


def test_when_predict_is_called_then_each_row_exposes_three_probability_fields():
    """AC1: Every prediction row has prob_home, prob_draw, prob_away attributes."""
    model = _fitted_model()
    X = pd.DataFrame({"elo_diff": [0.0, 100.0]})
    result = model.predict(X)
    assert len(result) == 2
    for row in result:
        assert hasattr(row, "prob_home"), "missing prob_home"
        assert hasattr(row, "prob_draw"), "missing prob_draw"
        assert hasattr(row, "prob_away"), "missing prob_away"


# ---------------------------------------------------------------------------
# AC2 — predict returns OutcomeProbabilities rows with probs summing to 1,
#         each probability in [0, 1]
# ---------------------------------------------------------------------------


def test_when_predict_returns_probabilities_then_they_sum_to_one():
    """AC2: prob_home + prob_draw + prob_away == 1.0 (within 1e-6) for every row."""
    model = _fitted_model()
    X = pd.DataFrame({"elo_diff": [-300.0, -100.0, 0.0, 100.0, 300.0]})
    result = model.predict(X)
    for row in result:
        total = row.prob_home + row.prob_draw + row.prob_away
        assert abs(total - 1.0) < 1e-6, f"probabilities sum to {total}, expected 1.0"


def test_when_predict_returns_probabilities_then_each_is_in_unit_interval():
    """AC2: Each of prob_home, prob_draw, prob_away is in [0, 1]."""
    model = _fitted_model()
    X = pd.DataFrame({"elo_diff": [-400.0, -200.0, 0.0, 200.0, 400.0]})
    result = model.predict(X)
    for row in result:
        assert 0.0 <= row.prob_home <= 1.0, f"prob_home out of [0,1]: {row.prob_home}"
        assert 0.0 <= row.prob_draw <= 1.0, f"prob_draw out of [0,1]: {row.prob_draw}"
        assert 0.0 <= row.prob_away <= 1.0, f"prob_away out of [0,1]: {row.prob_away}"


# Property: probability-sum invariant holds for ANY valid elo_diff input (AC2).
@given(
    elo_diffs=st.lists(
        st.floats(
            min_value=-1000.0,
            max_value=1000.0,
            allow_nan=False,
            allow_infinity=False,
        ),
        min_size=1,
        max_size=10,
    )
)
@settings(max_examples=20, deadline=None)
def test_when_any_valid_elo_diff_list_is_given_then_probabilities_sum_to_one(
    elo_diffs,
):
    """Property (AC2): probabilities sum to 1 for any non-empty list of valid elo_diff values."""
    model = _fitted_model()
    X = pd.DataFrame({"elo_diff": elo_diffs})
    result = model.predict(X)
    for row in result:
        total = row.prob_home + row.prob_draw + row.prob_away
        assert abs(total - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# AC3 — Monotonicity: larger positive elo_diff → larger prob_home
# ---------------------------------------------------------------------------


def test_when_elo_diff_increases_then_prob_home_increases():
    """AC3: predict(-200) has lower prob_home than predict(+200)."""
    model = _fitted_model()
    low = model.predict(pd.DataFrame({"elo_diff": [-200.0]}))
    high = model.predict(pd.DataFrame({"elo_diff": [200.0]}))
    assert high[0].prob_home > low[0].prob_home


def test_when_elo_diff_is_strongly_negative_then_prob_away_exceeds_prob_home():
    """AC3 (corollary): a large negative elo_diff flips the winner-probability."""
    model = _fitted_model()
    result = model.predict(pd.DataFrame({"elo_diff": [-500.0]}))
    assert result[0].prob_away > result[0].prob_home


# Property: monotonicity holds for any pair a < b within the valid range (AC3).
@given(
    a=st.floats(min_value=-800.0, max_value=-1.0, allow_nan=False, allow_infinity=False),
    b=st.floats(min_value=1.0, max_value=800.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=20, deadline=None)
def test_when_elo_diff_b_is_greater_than_a_then_prob_home_is_strictly_greater(a, b):
    """Property (AC3): for any a < b (with gap ≥ 2), prob_home(b) > prob_home(a)."""
    model = _fitted_model()
    result_a = model.predict(pd.DataFrame({"elo_diff": [a]}))
    result_b = model.predict(pd.DataFrame({"elo_diff": [b]}))
    assert result_a[0].prob_home < result_b[0].prob_home


# ---------------------------------------------------------------------------
# AC4 — Time-aware: fit trains on MatchDataset.train only — no internal
#         split/shuffle
# ---------------------------------------------------------------------------


def test_when_fit_is_called_then_train_property_is_accessed():
    """AC4: fit() must read .train from the dataset."""
    dataset = _make_dataset(n=200, seed=7)
    model = OrderedLogitModel()
    model.fit(cast(MatchDataset, dataset))
    assert dataset.train_accessed, "fit() must access .train"


def test_when_fit_is_called_then_test_property_is_never_accessed():
    """AC4: fit() must NOT read .test — time-aware, no internal split."""
    dataset = _make_dataset(n=200, seed=7)
    model = OrderedLogitModel()
    model.fit(cast(MatchDataset, dataset))
    assert not dataset.test_accessed, "fit() must NOT access .test; use .train only (time-aware)"


# ---------------------------------------------------------------------------
# AC5 — Deterministic: two fits on identical data give identical predictions
# ---------------------------------------------------------------------------


def test_when_fit_twice_on_same_data_then_predictions_are_identical():
    """AC5: BFGS over a convex objective is deterministic — same data → same result."""
    dataset = _make_dataset(n=200, seed=42)
    m1 = OrderedLogitModel()
    m1.fit(cast(MatchDataset, dataset))
    m2 = OrderedLogitModel()
    m2.fit(cast(MatchDataset, dataset))
    X = pd.DataFrame({"elo_diff": [-100.0, 0.0, 100.0]})
    r1, r2 = m1.predict(X), m2.predict(X)
    for a, b in zip(r1, r2):
        assert abs(a.prob_home - b.prob_home) < 1e-10
        assert abs(a.prob_draw - b.prob_draw) < 1e-10
        assert abs(a.prob_away - b.prob_away) < 1e-10


# ---------------------------------------------------------------------------
# Comment (SilvioBaratto) — finite coefficients on small but separable frame
# ---------------------------------------------------------------------------


def test_when_fit_on_small_separable_frame_then_all_probabilities_are_finite():
    """Guard degenerate edge: BFGS must yield finite probs on a small separable frame."""
    df = pd.DataFrame(
        {
            "elo_diff": [-300.0, -300.0, -200.0, 0.0, 0.0, 0.0, 200.0, 300.0, 300.0],
            "y_outcome": [0, 0, 0, 1, 1, 1, 2, 2, 2],
        }
    )
    dataset = _FakeMatchDataset(train_df=df, test_df=df.iloc[:0].copy())
    model = OrderedLogitModel()
    model.fit(cast(MatchDataset, dataset))
    X = pd.DataFrame({"elo_diff": [-100.0, 0.0, 100.0]})
    result = model.predict(X)
    for row in result:
        assert np.isfinite(row.prob_home), f"prob_home is not finite: {row.prob_home}"
        assert np.isfinite(row.prob_draw), f"prob_draw is not finite: {row.prob_draw}"
        assert np.isfinite(row.prob_away), f"prob_away is not finite: {row.prob_away}"


# ---------------------------------------------------------------------------
# AC6 — statsmodels added as a runtime dependency; no-key path works
# ---------------------------------------------------------------------------


def test_when_statsmodels_is_imported_then_it_is_available():
    """AC6: statsmodels must be importable (runtime dependency)."""
    mod = importlib.import_module("statsmodels")
    assert mod is not None


def test_when_no_api_key_is_set_then_fit_and_predict_complete_without_error(
    monkeypatch,
):
    """AC6: fit() and predict() work without FOOTBALL_DATA_API_KEY in the environment."""
    monkeypatch.delenv("FOOTBALL_DATA_API_KEY", raising=False)
    model = _fitted_model()
    X = pd.DataFrame({"elo_diff": [-50.0, 0.0, 50.0]})
    result = model.predict(X)
    assert len(result) == 3
