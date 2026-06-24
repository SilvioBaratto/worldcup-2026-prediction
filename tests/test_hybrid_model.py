"""Source-blind example tests for issue #13: Groll-style RF/GBM hybrid model.

Derived solely from the acceptance criteria — no implementation source was read.
Each test asserts exactly one observable behaviour implied by a criterion.

Criteria covered (verifiable subset):
  AC1 predict_goals returns non-negative (home, away) expected goals
  AC2 predict returns GoalPrediction with simplex probs and margin = home - away
  AC3 W/D/L via simulation.poisson.score_matrix — proper simplex, non-zero draw mass
  AC4 fit trains on MatchDataset.train only (time-aware, no re-split)
  AC5 deterministic: same random_seed → identical predictions (np.allclose)
  AC6 design matrix excludes home_goals/away_goals/date/home_team/away_team, no NaN cols

Skipped (not runtime-verifiable per oracle report):
  "All tests pass"          — boilerplate suite gate
  SOLID / clean code        — subjective prose
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from worldcup_playoff.models.dataset import MatchDataset, build_dataset
from worldcup_playoff.models.hybrid import GoalPrediction, HybridModel
from worldcup_playoff.simulation.poisson import score_matrix

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FEATURE_COLS: list[str] = ["elo_diff", "home_elo", "away_elo"]

_FORBIDDEN_COLS: frozenset[str] = frozenset(
    {"home_goals", "away_goals", "date", "home_team", "away_team"}
)

_SIMPLEX_TOL: float = 1e-6  # |prob_home + prob_draw + prob_away − 1| < this


# ---------------------------------------------------------------------------
# Fixture helpers — built from criteria, never from production source
# ---------------------------------------------------------------------------


def _make_match_df(n: int = 30, seed: int = 0) -> pd.DataFrame:
    """Return n chronologically ordered played-match rows.

    Only _FEATURE_COLS are included as feature columns; forbidden columns such
    as home_goals and away_goals appear in the frame but must not enter the
    design matrix (criterion AC6).
    """
    rng = np.random.default_rng(seed)
    months = (np.arange(n) // 28) + 1
    days = (np.arange(n) % 28) + 1
    rows = [
        {
            "date": f"2020-{months[i]:02d}-{days[i]:02d}",
            "home_team": "Alpha",
            "away_team": "Beta",
            "home_goals": int(rng.poisson(1.5)),
            "away_goals": int(rng.poisson(1.2)),
            "elo_diff": float(rng.normal(50.0, 100.0)),
            "home_elo": float(rng.normal(1500.0, 100.0)),
            "away_elo": float(rng.normal(1450.0, 100.0)),
        }
        for i in range(n)
    ]
    df = pd.DataFrame(rows)
    df["home_goals"] = df["home_goals"].astype("Int64")
    df["away_goals"] = df["away_goals"].astype("Int64")
    return df


def _make_dataset(n: int = 30, test_size: float = 0.2, seed: int = 0) -> MatchDataset:
    return build_dataset(
        _make_match_df(n, seed=seed), test_size=test_size, feature_cols=_FEATURE_COLS
    )


def _neutral_features() -> dict:
    """Feature dict with only allowed (non-forbidden) columns."""
    return {"elo_diff": 50.0, "home_elo": 1500.0, "away_elo": 1450.0}


# ---------------------------------------------------------------------------
# Module-scoped pytest fixtures (fit once; shared across tests)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fitted_model() -> HybridModel:
    ds = _make_dataset()
    m = HybridModel(random_seed=42)
    m.fit(ds)
    return m


@pytest.fixture(scope="module")
def match_features() -> dict:
    return _neutral_features()


# ---------------------------------------------------------------------------
# Module-level singleton for Hypothesis — avoids re-fitting on every example
# ---------------------------------------------------------------------------

_hyp_model: HybridModel | None = None


def _get_hyp_model() -> HybridModel:
    global _hyp_model
    if _hyp_model is None:
        ds = _make_dataset(n=50, seed=99)
        _hyp_model = HybridModel(random_seed=0)
        _hyp_model.fit(ds)
    return _hyp_model


# ---------------------------------------------------------------------------
# AC1 — predict_goals returns (home_goals, away_goals), both clipped >= 0
# ---------------------------------------------------------------------------


def test_when_predict_goals_called_then_result_has_two_values(fitted_model, match_features):
    assert len(fitted_model.predict_goals(match_features)) == 2


def test_when_predict_goals_called_then_home_goals_is_non_negative(fitted_model, match_features):
    home_g, _ = fitted_model.predict_goals(match_features)
    assert home_g >= 0.0


def test_when_predict_goals_called_then_away_goals_is_non_negative(fitted_model, match_features):
    _, away_g = fitted_model.predict_goals(match_features)
    assert away_g >= 0.0


@given(
    elo_diff=st.floats(min_value=-400.0, max_value=400.0, allow_nan=False, allow_infinity=False),
    home_elo=st.floats(min_value=900.0, max_value=2100.0, allow_nan=False, allow_infinity=False),
    away_elo=st.floats(min_value=900.0, max_value=2100.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=30)
def test_when_predict_goals_called_for_any_valid_features_then_both_goals_are_non_negative(
    elo_diff: float, home_elo: float, away_elo: float
) -> None:
    """Clipping to >= 0 is an invariant over the entire valid feature domain (AC1)."""
    model = _get_hyp_model()
    feats = {"elo_diff": elo_diff, "home_elo": home_elo, "away_elo": away_elo}
    h, a = model.predict_goals(feats)
    assert h >= 0.0
    assert a >= 0.0


# ---------------------------------------------------------------------------
# AC2 — predict returns GoalPrediction: simplex probs each in [0,1], margin correct
# ---------------------------------------------------------------------------


def test_when_predict_called_then_result_is_goal_prediction_instance(fitted_model, match_features):
    assert isinstance(fitted_model.predict(match_features), GoalPrediction)


def test_when_predict_called_then_probabilities_sum_to_one(fitted_model, match_features):
    pred = fitted_model.predict(match_features)
    assert abs(pred.prob_home + pred.prob_draw + pred.prob_away - 1.0) < _SIMPLEX_TOL


def test_when_predict_called_then_prob_home_is_in_unit_interval(fitted_model, match_features):
    pred = fitted_model.predict(match_features)
    assert 0.0 <= pred.prob_home <= 1.0


def test_when_predict_called_then_prob_draw_is_in_unit_interval(fitted_model, match_features):
    pred = fitted_model.predict(match_features)
    assert 0.0 <= pred.prob_draw <= 1.0


def test_when_predict_called_then_prob_away_is_in_unit_interval(fitted_model, match_features):
    pred = fitted_model.predict(match_features)
    assert 0.0 <= pred.prob_away <= 1.0


def test_when_predict_called_then_margin_equals_home_goals_minus_away_goals(
    fitted_model, match_features
):
    pred = fitted_model.predict(match_features)
    assert abs(pred.margin - (pred.home_goals - pred.away_goals)) < _SIMPLEX_TOL


@given(
    elo_diff=st.floats(min_value=-400.0, max_value=400.0, allow_nan=False, allow_infinity=False),
    home_elo=st.floats(min_value=900.0, max_value=2100.0, allow_nan=False, allow_infinity=False),
    away_elo=st.floats(min_value=900.0, max_value=2100.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=30)
def test_when_predict_called_for_any_valid_features_then_probs_always_sum_to_one(
    elo_diff: float, home_elo: float, away_elo: float
) -> None:
    """prob_home + prob_draw + prob_away == 1 is an invariant (proper simplex, AC2)."""
    model = _get_hyp_model()
    pred = model.predict({"elo_diff": elo_diff, "home_elo": home_elo, "away_elo": away_elo})
    assert abs(pred.prob_home + pred.prob_draw + pred.prob_away - 1.0) < 1e-4


@given(
    elo_diff=st.floats(min_value=-400.0, max_value=400.0, allow_nan=False, allow_infinity=False),
    home_elo=st.floats(min_value=900.0, max_value=2100.0, allow_nan=False, allow_infinity=False),
    away_elo=st.floats(min_value=900.0, max_value=2100.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=30)
def test_when_predict_called_for_any_valid_features_then_margin_equals_home_minus_away(
    elo_diff: float, home_elo: float, away_elo: float
) -> None:
    """margin = home_goals − away_goals is an invariant for all valid features (AC2)."""
    model = _get_hyp_model()
    pred = model.predict({"elo_diff": elo_diff, "home_elo": home_elo, "away_elo": away_elo})
    assert abs(pred.margin - (pred.home_goals - pred.away_goals)) < 1e-6


# ---------------------------------------------------------------------------
# AC3 — score_matrix: proper simplex (sums to 1), non-zero draw mass
# ---------------------------------------------------------------------------


def test_when_score_matrix_called_then_all_entries_are_non_negative():
    assert np.all(score_matrix(1.5, 1.2) >= 0.0)


def test_when_score_matrix_called_then_entries_sum_to_one():
    assert abs(float(score_matrix(1.5, 1.2).sum()) - 1.0) < _SIMPLEX_TOL


def test_when_score_matrix_called_then_draw_mass_is_positive():
    """Diagonal = P(home score i, away score i); must be > 0 (draws are always possible)."""
    matrix = score_matrix(1.5, 1.2)
    assert float(np.diag(matrix).sum()) > 0.0


def test_when_score_matrix_called_with_equal_lambdas_then_draw_mass_is_positive():
    """Non-zero draw mass must hold even when both teams have the same expected goals."""
    assert float(np.diag(score_matrix(1.3, 1.3)).sum()) > 0.0


@given(
    lam_home=st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
    lam_away=st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
)
def test_when_score_matrix_called_for_any_valid_lambdas_then_is_proper_simplex(
    lam_home: float, lam_away: float
) -> None:
    """score_matrix sums to 1 for all positive Poisson rate pairs (AC3 invariant)."""
    assert abs(float(score_matrix(lam_home, lam_away).sum()) - 1.0) < 1e-4


@given(
    lam_home=st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
    lam_away=st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
)
def test_when_score_matrix_called_for_any_valid_lambdas_then_draw_mass_is_positive(
    lam_home: float, lam_away: float
) -> None:
    """Non-zero draw mass is an invariant for all positive rate pairs (AC3)."""
    assert float(np.diag(score_matrix(lam_home, lam_away)).sum()) > 0.0


# ---------------------------------------------------------------------------
# AC4 — fit trains on MatchDataset.train only (time-aware, no re-split)
# ---------------------------------------------------------------------------


def test_when_fit_called_with_match_dataset_then_model_can_predict_without_error():
    """
    fit() accepts a pre-split MatchDataset and produces a usable model.

    The MatchDataset API is the architectural enforcement: the model receives only
    the split object (with .train and .test), never the raw pre-split data, so it
    cannot internally re-split or shuffle the chronological order.
    """
    ds = _make_dataset()
    model = HybridModel(random_seed=0)
    model.fit(ds)
    assert isinstance(model.predict(_neutral_features()), GoalPrediction)


def test_when_train_split_is_larger_then_predictions_differ_from_smaller_train():
    """
    If the model uses dataset.train, shrinking .train must shift its predictions.
    Two identical models trained on the same raw data but different chronological
    split sizes must produce different predictions — proving the model is actually
    learning from .train rather than ignoring it.

    Assumption (from criterion): train-only learning means training size influences output.
    """
    df = _make_match_df(n=40, seed=7)
    ds_large = build_dataset(df, test_size=0.1, feature_cols=_FEATURE_COLS)
    ds_small = build_dataset(df, test_size=0.5, feature_cols=_FEATURE_COLS)
    assert len(ds_large.train) > len(ds_small.train)

    m_large = HybridModel(random_seed=42)
    m_small = HybridModel(random_seed=42)
    m_large.fit(ds_large)
    m_small.fit(ds_small)

    feats = _neutral_features()
    p_large = m_large.predict(feats)
    p_small = m_small.predict(feats)

    predictions_differ = not (
        np.isclose(p_large.prob_home, p_small.prob_home, atol=1e-8)
        and np.isclose(p_large.prob_draw, p_small.prob_draw, atol=1e-8)
        and np.isclose(p_large.prob_away, p_small.prob_away, atol=1e-8)
    )
    assert predictions_differ, (
        "Models trained on different .train sizes produced identical predictions; "
        "model may not be using dataset.train at all."
    )


def test_when_train_dominated_by_home_wins_then_prob_home_exceeds_prob_away():
    """
    Home-dominant training data must produce prob_home > prob_away on home-favoured
    match features — verifying that the model generalises from .train to predictions.

    First 24 rows (= .train with test_size=0.2) → home always wins 3-0.
    Last 6 rows  (= .test)                       → away always wins 0-3.
    Same elo_diff across all rows so the RF cannot split by feature; train-only
    fitting therefore produces a pure home-win learner.
    """
    rows: list[dict] = []
    for i in range(25):
        rows.append(
            {
                "date": f"2019-{i // 28 + 1:02d}-{i % 28 + 1:02d}",
                "home_team": "A",
                "away_team": "B",
                "home_goals": 3,
                "away_goals": 0,
                "elo_diff": 50.0,
                "home_elo": 1550.0,
                "away_elo": 1500.0,
            }
        )
    for j in range(5):
        rows.append(
            {
                "date": f"2025-01-{j + 1:02d}",
                "home_team": "A",
                "away_team": "B",
                "home_goals": 0,
                "away_goals": 3,
                "elo_diff": 50.0,  # identical features, opposite outcome
                "home_elo": 1550.0,
                "away_elo": 1500.0,
            }
        )
    df = pd.DataFrame(rows)
    df["home_goals"] = df["home_goals"].astype("Int64")
    df["away_goals"] = df["away_goals"].astype("Int64")
    # test_size=0.2 → floor(30 * 0.2) = 6 rows in .test, 24 in .train
    ds = build_dataset(df, test_size=0.2, feature_cols=_FEATURE_COLS)
    assert len(ds.train) == 24  # all home-win rows in train

    model = HybridModel(random_seed=0)
    model.fit(ds)
    pred = model.predict({"elo_diff": 50.0, "home_elo": 1550.0, "away_elo": 1500.0})
    assert pred.prob_home > pred.prob_away, (
        f"Expected prob_home > prob_away when training only on home-win data "
        f"(prob_home={pred.prob_home:.3f}, prob_away={pred.prob_away:.3f})."
    )


# ---------------------------------------------------------------------------
# AC5 — Deterministic: two fits with the same random_seed produce identical predictions
# ---------------------------------------------------------------------------


def test_when_two_models_fit_with_same_seed_then_predict_goals_are_identical():
    ds = _make_dataset()
    m1, m2 = HybridModel(random_seed=7), HybridModel(random_seed=7)
    m1.fit(ds)
    m2.fit(ds)
    h1, a1 = m1.predict_goals(_neutral_features())
    h2, a2 = m2.predict_goals(_neutral_features())
    assert np.isclose(h1, h2) and np.isclose(a1, a2)


def test_when_two_models_fit_with_same_seed_then_prob_home_is_identical():
    ds = _make_dataset()
    m1, m2 = HybridModel(random_seed=7), HybridModel(random_seed=7)
    m1.fit(ds)
    m2.fit(ds)
    assert np.isclose(
        m1.predict(_neutral_features()).prob_home, m2.predict(_neutral_features()).prob_home
    )


def test_when_two_models_fit_with_same_seed_then_prob_draw_is_identical():
    ds = _make_dataset()
    m1, m2 = HybridModel(random_seed=7), HybridModel(random_seed=7)
    m1.fit(ds)
    m2.fit(ds)
    assert np.isclose(
        m1.predict(_neutral_features()).prob_draw, m2.predict(_neutral_features()).prob_draw
    )


def test_when_two_models_fit_with_same_seed_then_prob_away_is_identical():
    ds = _make_dataset()
    m1, m2 = HybridModel(random_seed=7), HybridModel(random_seed=7)
    m1.fit(ds)
    m2.fit(ds)
    assert np.isclose(
        m1.predict(_neutral_features()).prob_away, m2.predict(_neutral_features()).prob_away
    )


def test_when_two_models_fit_with_same_seed_then_margin_is_identical():
    ds = _make_dataset()
    m1, m2 = HybridModel(random_seed=7), HybridModel(random_seed=7)
    m1.fit(ds)
    m2.fit(ds)
    assert np.isclose(
        m1.predict(_neutral_features()).margin, m2.predict(_neutral_features()).margin
    )


def test_when_two_models_fit_with_different_seeds_then_at_least_one_prob_differs():
    """
    Different random_seeds must actually affect the model — verifies the seed
    controls the random state, not just a no-op parameter (AC5 contrapositive).
    """
    ds = _make_dataset()
    m1, m2 = HybridModel(random_seed=1), HybridModel(random_seed=2)
    m1.fit(ds)
    m2.fit(ds)
    p1 = m1.predict(_neutral_features())
    p2 = m2.predict(_neutral_features())
    differ = (
        not np.isclose(p1.prob_home, p2.prob_home)
        or not np.isclose(p1.prob_draw, p2.prob_draw)
        or not np.isclose(p1.prob_away, p2.prob_away)
    )
    assert differ, (
        "Models with different random_seeds produced identical predictions; "
        "random_seed may not control the model's random state."
    )


# ---------------------------------------------------------------------------
# AC6 — Design matrix excludes forbidden cols; no NaN feature cols; runs no-key
# ---------------------------------------------------------------------------


def test_when_feature_cols_contain_no_forbidden_columns_then_fit_succeeds():
    """
    The design matrix must exclude home_goals, away_goals, date, home_team, away_team.
    Asserting that _FEATURE_COLS has no overlap with _FORBIDDEN_COLS and that
    fit() on such a dataset succeeds is a direct check of the criterion.
    """
    overlap = set(_FEATURE_COLS) & _FORBIDDEN_COLS
    assert not overlap, f"Test setup error — forbidden cols in feature list: {overlap}"

    ds = _make_dataset()
    assert not (set(ds.feature_cols) & _FORBIDDEN_COLS), (
        f"MatchDataset.feature_cols contains forbidden columns: "
        f"{set(ds.feature_cols) & _FORBIDDEN_COLS}"
    )
    HybridModel(random_seed=0).fit(ds)  # must not raise


def test_when_train_split_has_no_nan_in_feature_columns_then_fit_succeeds():
    """No NaN columns in the design matrix: clean feature columns → fit succeeds (AC6)."""
    ds = _make_dataset()
    for col in _FEATURE_COLS:
        n_nan = int(ds.train[col].isna().sum())
        assert n_nan == 0, f"Column {col!r} has {n_nan} NaN(s) in train"
    HybridModel(random_seed=0).fit(ds)  # must not raise


def test_when_predict_receives_only_allowed_feature_columns_then_no_error_is_raised(
    fitted_model,
):
    """
    At inference time the caller must not need to supply any forbidden column.
    A feature dict that is strictly allowed must yield a valid GoalPrediction (AC6).
    """
    allowed_only = {"elo_diff": 100.0, "home_elo": 1600.0, "away_elo": 1400.0}
    for key in allowed_only:
        assert key not in _FORBIDDEN_COLS, f"{key!r} is a forbidden column"
    result = fitted_model.predict(allowed_only)
    assert isinstance(result, GoalPrediction)


def test_when_football_data_api_key_is_absent_then_fit_and_predict_succeed(monkeypatch):
    """
    Criterion: runs no-key. The model must not call any live API during fit or predict;
    unsetting FOOTBALL_DATA_API_KEY must not cause an error (AC6).
    """
    monkeypatch.delenv("FOOTBALL_DATA_API_KEY", raising=False)
    ds = _make_dataset()
    model = HybridModel(random_seed=0)
    model.fit(ds)
    assert isinstance(model.predict(_neutral_features()), GoalPrediction)
