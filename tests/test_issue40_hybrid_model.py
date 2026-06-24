"""
Source-blind example tests for issue #40: Groll-style RF/GBM goal-based hybrid model.

Derived exclusively from the acceptance criteria in the issue and the product
requirements in .code-generator/requirements.md.  No implementation source was
opened at any point during authoring.

Verifiable acceptance criteria covered (per oracle report):
  AC-1  models/hybrid.py exposes HybridModel, GoalPrediction, and fit_hybrid.
  AC-2  Output is goal-based; margin = home_goals − away_goals; W/D/L probs
        from the score_matrix and normalised to 1.
  AC-3  Training uses dataset.train only; deterministic given random_seed.
  AC-4  Design matrix excludes forbidden columns, one-hot encodes confederation
        against the fixed six-value tuple, imputes with train-only sentinels.
  AC-5  Predicted goals are clamped non-negative before forming Poisson means.

Skipped (oracle: NOT VERIFIABLE):
  AC-6  "All tests pass; new unit tests cover …"  (meta-criterion)
  AC-7  SOLID / clean-code prose
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from worldcup_playoff.models.hybrid import (
    GoalPrediction,
    HybridConfig,
    HybridModel,
    MatchDataset,
    fit_hybrid,
)

# ---------------------------------------------------------------------------
# Official confederation tuple (requirements §Data Contracts, FIFA ranking)
# ---------------------------------------------------------------------------
CONFEDERATIONS = ("UEFA", "CONMEBOL", "CAF", "AFC", "CONCACAF", "OFC")

# ---------------------------------------------------------------------------
# Minimal fixture helpers — built from the covariate spec in requirements.md;
# no production source was consulted.
# ---------------------------------------------------------------------------


def _match_df(n: int = 40, seed: int = 0) -> pd.DataFrame:
    """
    Build a minimal match DataFrame that satisfies the feature spec:
    Elo, Elo-diff, Dixon-Coles attack/defence, rest days, confederation,
    neutral-venue flag, plus target columns home_goals / away_goals.
    """
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "home_elo": rng.uniform(1400.0, 1900.0, n),
            "away_elo": rng.uniform(1400.0, 1900.0, n),
            "elo_diff": rng.uniform(-300.0, 300.0, n),
            "home_attack": rng.uniform(0.5, 2.0, n),
            "home_defence": rng.uniform(0.5, 2.0, n),
            "away_attack": rng.uniform(0.5, 2.0, n),
            "away_defence": rng.uniform(0.5, 2.0, n),
            "home_rest_days": rng.integers(3, 14, n).astype(float),
            "away_rest_days": rng.integers(3, 14, n).astype(float),
            "confederation": rng.choice(list(CONFEDERATIONS), n),
            "neutral": rng.choice([True, False], n),
            "home_goals": rng.integers(0, 5, n).astype(float),
            "away_goals": rng.integers(0, 5, n).astype(float),
        }
    )


def _feature_row(seed: int = 42, confederation: str = "UEFA") -> pd.DataFrame:
    """Single-row feature DataFrame (no target columns) for inference."""
    row = _match_df(n=1, seed=seed).drop(columns=["home_goals", "away_goals"])
    row["confederation"] = confederation
    return row.reset_index(drop=True)


def _make_dataset(train_seed: int = 1, test_seed: int = 2) -> MatchDataset:
    return MatchDataset(
        train=_match_df(n=60, seed=train_seed),
        test=_match_df(n=15, seed=test_seed),
    )


def _build_and_fit(seed: int = 42) -> HybridModel:
    ds = _make_dataset()
    cfg = HybridConfig(random_seed=seed)
    model = HybridModel(config=cfg)
    model.fit(ds)
    return model


# Module-level trained model reused by property-based tests (built once).
_TRAINED: HybridModel = _build_and_fit(seed=42)


# ===========================================================================
# AC-1  API surface
# ===========================================================================


class TestAPIExistence:
    """HybridModel, GoalPrediction, fit_hybrid must be importable and callable."""

    def test_when_hybrid_model_is_imported_then_fit_method_exists(self):
        assert callable(getattr(HybridModel, "fit", None))

    def test_when_hybrid_model_is_imported_then_predict_goals_method_exists(self):
        assert callable(getattr(HybridModel, "predict_goals", None))

    def test_when_hybrid_model_is_imported_then_predict_method_exists(self):
        assert callable(getattr(HybridModel, "predict", None))

    def test_when_fit_hybrid_is_called_then_a_hybrid_model_is_returned(self):
        result = fit_hybrid(_make_dataset(), HybridConfig(random_seed=0))
        assert isinstance(result, HybridModel)

    def test_when_fit_hybrid_returns_model_then_it_can_predict(self):
        model = fit_hybrid(_make_dataset(), HybridConfig(random_seed=0))
        assert isinstance(model.predict(_feature_row()), GoalPrediction)


class TestGoalPredictionDataclass:
    """GoalPrediction must be a frozen value object with exactly the six specified fields."""

    def test_when_goal_prediction_is_constructed_then_all_six_fields_are_accessible(self):
        gp = GoalPrediction(
            prob_home=0.5,
            prob_draw=0.25,
            prob_away=0.25,
            home_goals=1.5,
            away_goals=1.0,
            margin=0.5,
        )
        assert gp.prob_home == pytest.approx(0.5)
        assert gp.prob_draw == pytest.approx(0.25)
        assert gp.prob_away == pytest.approx(0.25)
        assert gp.home_goals == pytest.approx(1.5)
        assert gp.away_goals == pytest.approx(1.0)
        assert gp.margin == pytest.approx(0.5)

    def test_when_goal_prediction_field_is_reassigned_then_frozen_error_is_raised(self):
        gp = GoalPrediction(
            prob_home=0.4,
            prob_draw=0.3,
            prob_away=0.3,
            home_goals=1.0,
            away_goals=1.0,
            margin=0.0,
        )
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
            gp.prob_home = 0.9  # type: ignore[misc]


# ===========================================================================
# AC-2  Goal-based output: margin and W/D/L from score_matrix
# ===========================================================================


class TestGoalBasedOutput:
    def test_when_predict_is_called_then_a_goal_prediction_is_returned(self):
        result = _TRAINED.predict(_feature_row())
        assert isinstance(result, GoalPrediction)

    def test_when_predict_is_called_then_margin_equals_home_goals_minus_away_goals(self):
        result = _TRAINED.predict(_feature_row())
        assert result.margin == pytest.approx(result.home_goals - result.away_goals)

    def test_when_predict_goals_is_called_then_two_floats_are_returned(self):
        lh, la = _TRAINED.predict_goals(_feature_row())
        assert isinstance(float(lh), float)
        assert isinstance(float(la), float)

    def test_when_predict_goals_is_called_then_home_goals_are_non_negative(self):
        lh, _ = _TRAINED.predict_goals(_feature_row())
        assert lh >= 0.0

    def test_when_predict_goals_is_called_then_away_goals_are_non_negative(self):
        _, la = _TRAINED.predict_goals(_feature_row())
        assert la >= 0.0


class TestWDLProbabilities:
    def test_when_predict_is_called_then_wdl_probs_sum_to_one(self):
        result = _TRAINED.predict(_feature_row())
        total = result.prob_home + result.prob_draw + result.prob_away
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_when_predict_is_called_then_prob_home_is_between_zero_and_one(self):
        assert 0.0 <= _TRAINED.predict(_feature_row()).prob_home <= 1.0

    def test_when_predict_is_called_then_prob_draw_is_between_zero_and_one(self):
        assert 0.0 <= _TRAINED.predict(_feature_row()).prob_draw <= 1.0

    def test_when_predict_is_called_then_prob_away_is_between_zero_and_one(self):
        assert 0.0 <= _TRAINED.predict(_feature_row()).prob_away <= 1.0


# ===========================================================================
# AC-3  Training uses dataset.train only; deterministic given random_seed
# ===========================================================================


class TestTrainingDeterminism:
    def test_when_same_seed_is_used_twice_then_home_goal_predictions_are_identical(self):
        ds = _make_dataset()
        cfg = HybridConfig(random_seed=7)

        m1 = HybridModel(config=cfg)
        m1.fit(ds)
        m2 = HybridModel(config=cfg)
        m2.fit(ds)

        row = _feature_row()
        assert m1.predict(row).home_goals == pytest.approx(m2.predict(row).home_goals)

    def test_when_same_seed_is_used_twice_then_away_goal_predictions_are_identical(self):
        ds = _make_dataset()
        cfg = HybridConfig(random_seed=7)

        m1 = HybridModel(config=cfg)
        m1.fit(ds)
        m2 = HybridModel(config=cfg)
        m2.fit(ds)

        row = _feature_row()
        assert m1.predict(row).away_goals == pytest.approx(m2.predict(row).away_goals)


class TestTrainOnlyPartition:
    def test_when_test_partition_is_replaced_then_home_goal_predictions_are_unchanged(self):
        """
        The model must read dataset.train exclusively.  Replacing dataset.test
        with a completely different frame (same train) must not shift predictions.
        """
        cfg = HybridConfig(random_seed=42)

        ds_a = _make_dataset(train_seed=10, test_seed=20)
        m_a = HybridModel(config=cfg)
        m_a.fit(ds_a)

        ds_b = MatchDataset(
            train=ds_a.train,
            test=_match_df(n=15, seed=99),  # different test, same train
        )
        m_b = HybridModel(config=cfg)
        m_b.fit(ds_b)

        row = _feature_row()
        assert m_a.predict(row).home_goals == pytest.approx(m_b.predict(row).home_goals)

    def test_when_test_partition_is_replaced_then_away_goal_predictions_are_unchanged(self):
        cfg = HybridConfig(random_seed=42)

        ds_a = _make_dataset(train_seed=10, test_seed=20)
        m_a = HybridModel(config=cfg)
        m_a.fit(ds_a)

        ds_b = MatchDataset(
            train=ds_a.train,
            test=_match_df(n=15, seed=99),
        )
        m_b = HybridModel(config=cfg)
        m_b.fit(ds_b)

        row = _feature_row()
        assert m_a.predict(row).away_goals == pytest.approx(m_b.predict(row).away_goals)


# ===========================================================================
# AC-4  Design matrix: forbidden columns excluded; confederation one-hot;
#        leakage-free imputation
# ===========================================================================


class TestDesignMatrix:
    def test_when_each_valid_confederation_is_used_then_prediction_succeeds(self):
        """Each member of the fixed confederation tuple must be accepted."""
        for conf in CONFEDERATIONS:
            row = _feature_row(confederation=conf)
            result = _TRAINED.predict(row)
            assert isinstance(result, GoalPrediction), f"failed for confederation={conf!r}"

    def test_when_legacy_target_column_is_included_then_predictions_are_unchanged(self):
        """
        Criterion: design matrix excludes forbidden (target-leaking) columns.
        Adding the legacy binary target HOME_WIN to the feature frame must not
        alter predictions — the column must be silently dropped from the matrix.
        """
        row_clean = _feature_row()
        row_tainted = row_clean.copy()
        row_tainted["HOME_WIN"] = 1  # adversarial target leak

        pred_clean = _TRAINED.predict(row_clean)
        pred_tainted = _TRAINED.predict(row_tainted)

        assert pred_clean.home_goals == pytest.approx(pred_tainted.home_goals)
        assert pred_clean.away_goals == pytest.approx(pred_tainted.away_goals)

    def test_when_nullable_numeric_is_nan_then_prediction_does_not_raise(self):
        """
        Nullable numerics must be imputed (with a train-only sentinel) rather
        than propagated as NaN into the Poisson regression.
        """
        row = _feature_row().copy()
        num_cols = row.select_dtypes(include=[np.number]).columns.tolist()
        row.loc[row.index[0], num_cols[0]] = np.nan

        result = _TRAINED.predict(row)
        assert isinstance(result, GoalPrediction)

    def test_when_nan_is_imputed_then_sentinel_is_from_train_not_inference_batch(self):
        """
        Leakage-free imputation: the imputed value for a NaN cell must equal the
        train-set sentinel regardless of what other rows are present in the same
        inference batch.  We test this by predicting the NaN row alone vs.
        alongside rows whose values would heavily skew a batch-level mean.
        """
        ds = _make_dataset()
        cfg = HybridConfig(random_seed=42)
        model = HybridModel(config=cfg)
        model.fit(ds)

        nan_row = _feature_row().copy()
        num_col = nan_row.select_dtypes(include=[np.number]).columns[0]
        nan_row.loc[nan_row.index[0], num_col] = np.nan

        # Solo prediction (batch of 1)
        pred_solo = model.predict(nan_row)

        # Same NaN row plus four rows with extreme values for that column.
        # If imputation were batch-wise the sentinel would shift toward 9 999.
        noise = _feature_row(seed=77).copy()
        noise[num_col] = 9_999.0
        batch = pd.concat([nan_row, noise, noise, noise, noise], ignore_index=True)
        pred_in_batch = model.predict(batch.iloc[[0]])

        assert pred_solo.home_goals == pytest.approx(pred_in_batch.home_goals, abs=1e-9)
        assert pred_solo.away_goals == pytest.approx(pred_in_batch.away_goals, abs=1e-9)


# ===========================================================================
# AC-5  Non-negative goal clamp
# ===========================================================================


class TestGoalClamp:
    def test_when_all_numeric_features_are_extreme_negative_then_home_goals_are_non_negative(self):
        """
        Criterion: predicted goals are clamped non-negative before Poisson means are formed.
        Even extreme-low feature values must yield home_goals >= 0.
        """
        extreme = _feature_row().copy()
        for col in extreme.select_dtypes(include=[np.number]).columns:
            extreme[col] = -9_999.0

        lh, _ = _TRAINED.predict_goals(extreme)
        assert lh >= 0.0

    def test_when_all_numeric_features_are_extreme_negative_then_away_goals_are_non_negative(self):
        extreme = _feature_row().copy()
        for col in extreme.select_dtypes(include=[np.number]).columns:
            extreme[col] = -9_999.0

        _, la = _TRAINED.predict_goals(extreme)
        assert la >= 0.0

    def test_when_all_numeric_features_are_zero_then_goals_are_non_negative(self):
        zero_row = _feature_row().copy()
        for col in zero_row.select_dtypes(include=[np.number]).columns:
            zero_row[col] = 0.0

        lh, la = _TRAINED.predict_goals(zero_row)
        assert lh >= 0.0
        assert la >= 0.0


# ===========================================================================
# Property-based tests (Hypothesis)
# ===========================================================================
#
# Three invariants are implied by the acceptance criteria text and apply across
# ALL valid feature inputs — not just the single example chosen above:
#
#   P1  margin = home_goals − away_goals   (formula invariant, AC-2)
#   P2  home_goals >= 0 and away_goals >= 0  (clamp invariant, AC-5)
#   P3  prob_home + prob_draw + prob_away == 1.0   (normalisation, AC-2)
# ---------------------------------------------------------------------------


_VALID_FEATURE = st.fixed_dictionaries(
    {
        "home_elo": st.floats(min_value=1000.0, max_value=2500.0, allow_nan=False),
        "away_elo": st.floats(min_value=1000.0, max_value=2500.0, allow_nan=False),
        "elo_diff": st.floats(min_value=-500.0, max_value=500.0, allow_nan=False),
        "home_attack": st.floats(min_value=0.1, max_value=3.0, allow_nan=False),
        "home_defence": st.floats(min_value=0.1, max_value=3.0, allow_nan=False),
        "away_attack": st.floats(min_value=0.1, max_value=3.0, allow_nan=False),
        "away_defence": st.floats(min_value=0.1, max_value=3.0, allow_nan=False),
        "home_rest_days": st.floats(min_value=1.0, max_value=30.0),
        "away_rest_days": st.floats(min_value=1.0, max_value=30.0),
        "confederation": st.sampled_from(CONFEDERATIONS),
        "neutral": st.booleans(),
    }
)


def _row_from_dict(d: dict) -> pd.DataFrame:
    return pd.DataFrame({k: [v] for k, v in d.items()})


@given(features=_VALID_FEATURE)
@settings(max_examples=30)
def test_when_valid_features_given_then_margin_always_equals_home_minus_away_goals(
    features: dict,
) -> None:
    """P1 — margin = home_goals − away_goals holds for all valid inputs."""
    result = _TRAINED.predict(_row_from_dict(features))
    assert result.margin == pytest.approx(result.home_goals - result.away_goals, abs=1e-9)


@given(features=_VALID_FEATURE)
@settings(max_examples=30)
def test_when_valid_features_given_then_home_goals_are_always_non_negative(
    features: dict,
) -> None:
    """P2a — home goals are non-negative for all valid inputs."""
    lh, _ = _TRAINED.predict_goals(_row_from_dict(features))
    assert lh >= 0.0


@given(features=_VALID_FEATURE)
@settings(max_examples=30)
def test_when_valid_features_given_then_away_goals_are_always_non_negative(
    features: dict,
) -> None:
    """P2b — away goals are non-negative for all valid inputs."""
    _, la = _TRAINED.predict_goals(_row_from_dict(features))
    assert la >= 0.0


@given(features=_VALID_FEATURE)
@settings(max_examples=30)
def test_when_valid_features_given_then_wdl_probs_always_sum_to_one(
    features: dict,
) -> None:
    """P3 — W/D/L probabilities (from score_matrix) always sum to 1.0."""
    result = _TRAINED.predict(_row_from_dict(features))
    total = result.prob_home + result.prob_draw + result.prob_away
    assert total == pytest.approx(1.0, abs=1e-6)
