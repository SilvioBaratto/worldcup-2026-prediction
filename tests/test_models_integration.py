"""End-to-end integration tests for Issue #15.

Covers every verifiable acceptance criterion for the models-package
no-key smoke test (hybrid + ordered logit + baseline).

  AC1  No-key pipeline: martj42 fixture →
         compute_elo → fit_dixon_coles → build_features → build_dataset
         → fit both new models successfully.
  AC2  Hybrid output: predicted goals >= 0 and W/D/L probs sum to 1.
  AC3  Ordered-logit output: W/D/L probs sum to 1.
  AC4  Determinism: same seed → identical predictions (both models).
  AC5  ClassifierFactory.create still builds svm / random_forest / naive_bayes.
  AC6  worldcup_playoff.models re-exports the full Cycle-4 surface
       alongside the legacy exports.

No football-data.org API key is required: the pipeline uses only martj42
CC0 history (in-memory fixture) + computed Elo/Poisson features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from worldcup_playoff.config import TrainingConfig
from worldcup_playoff.data.elo import compute_elo
from worldcup_playoff.features.build import build_features
from worldcup_playoff.models.classifiers import ClassifierFactory
from worldcup_playoff.models.dataset import MatchDataset, add_targets, build_dataset
from worldcup_playoff.models.hybrid import GoalPrediction, HybridModel
from worldcup_playoff.models.ordered_logit import OrderedLogitModel
from worldcup_playoff.simulation.poisson import fit_dixon_coles

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Numeric feature columns present in build_features output; confederation
# columns are excluded here to keep the fixture simple.
_FEATURE_COLS: list[str] = [
    "elo_diff",
    "home_elo",
    "away_elo",
    "home_attack",
    "home_defence",
    "away_attack",
    "away_defence",
    "neutral",
]

# A representative single-match feature dict for HybridModel.predict
_SAMPLE_FEATS: dict = {
    "elo_diff": 50.0,
    "home_elo": 1550.0,
    "away_elo": 1500.0,
    "home_attack": 0.1,
    "home_defence": -0.1,
    "away_attack": 0.0,
    "away_defence": 0.0,
    "neutral": True,
}

_TEAMS = ["Brazil", "Argentina", "France", "Germany", "Spain", "England", "Italy", "Portugal"]

_SIMPLEX_TOL: float = 1e-6


# ---------------------------------------------------------------------------
# In-memory martj42 fixture (UPPERCASE column schema — no network required)
# ---------------------------------------------------------------------------


def _make_martj42(n: int = 48) -> pd.DataFrame:
    """Return n rows in martj42 UPPERCASE schema; all matches have known scores.

    Dates monotonically increase (30 per year) so chronological_split is
    well-defined.  home_team != away_team by construction (offset by 1 in
    the 8-team cycle).
    """
    rows = [
        {
            "DATE": pd.Timestamp(f"20{18 + i // 30:02d}-01-{i % 30 + 1:02d}"),
            "HOME_TEAM": _TEAMS[i % len(_TEAMS)],
            "AWAY_TEAM": _TEAMS[(i + 1) % len(_TEAMS)],
            "HOME_GOALS": i % 4,
            "AWAY_GOALS": i % 3,
            "TOURNAMENT": "FIFA World Cup" if i % 7 == 0 else "Friendly",
            "NEUTRAL": i % 5 == 0,
        }
        for i in range(n)
    ]
    df = pd.DataFrame(rows)
    df["HOME_GOALS"] = df["HOME_GOALS"].astype("Int64")
    df["AWAY_GOALS"] = df["AWAY_GOALS"].astype("Int64")
    return df


def _build_dataset(n: int = 48, test_size: float = 0.2) -> MatchDataset:
    """Run the full no-key pipeline and return a MatchDataset."""
    df = _make_martj42(n)
    elo = compute_elo(df)
    abilities = fit_dixon_coles(df)
    features = add_targets(build_features(df, elo, abilities))
    return build_dataset(features, test_size=test_size, feature_cols=_FEATURE_COLS)


# ---------------------------------------------------------------------------
# Module-scoped fixtures (pipeline runs once; models shared across tests)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pipeline_dataset() -> MatchDataset:
    """MatchDataset produced by the full no-key pipeline (shared for speed)."""
    return _build_dataset()


@pytest.fixture(scope="module")
def fitted_hybrid(pipeline_dataset: MatchDataset) -> HybridModel:
    return HybridModel(random_seed=42).fit(pipeline_dataset)


@pytest.fixture(scope="module")
def fitted_logit(pipeline_dataset: MatchDataset) -> OrderedLogitModel:
    return OrderedLogitModel().fit(pipeline_dataset)


# ---------------------------------------------------------------------------
# AC1 — no-key end-to-end pipeline
# ---------------------------------------------------------------------------


def test_when_martj42_is_piped_through_full_pipeline_then_hybrid_model_can_predict(
    fitted_hybrid: HybridModel,
) -> None:
    """AC1: the full no-key pipeline yields a HybridModel that returns GoalPrediction."""
    pred = fitted_hybrid.predict(_SAMPLE_FEATS)
    assert isinstance(pred, GoalPrediction)


def test_when_martj42_is_piped_through_full_pipeline_then_ordered_logit_can_predict(
    fitted_logit: OrderedLogitModel,
) -> None:
    """AC1: the full no-key pipeline yields an OrderedLogitModel that predicts one row."""
    result = fitted_logit.predict(pd.DataFrame({"elo_diff": [0.0]}))
    assert len(result) == 1


def test_when_pipeline_is_run_then_dataset_has_both_train_and_test_splits(
    pipeline_dataset: MatchDataset,
) -> None:
    """AC1: build_dataset produces non-empty train and test splits."""
    assert len(pipeline_dataset.train) > 0
    assert len(pipeline_dataset.test) > 0


def test_when_pipeline_is_run_then_dataset_contains_elo_diff_feature(
    pipeline_dataset: MatchDataset,
) -> None:
    """AC1: elo_diff (needed by ordered logit) is present in the train split."""
    assert "elo_diff" in pipeline_dataset.train.columns


def test_when_pipeline_is_run_then_dataset_contains_y_outcome_target(
    pipeline_dataset: MatchDataset,
) -> None:
    """AC1: y_outcome (needed by ordered logit) is present after add_targets."""
    assert "y_outcome" in pipeline_dataset.train.columns


# ---------------------------------------------------------------------------
# AC2 — Hybrid output: goals >= 0, W/D/L probs sum to 1
# ---------------------------------------------------------------------------


def test_when_hybrid_predicts_goals_then_home_goals_are_non_negative(
    fitted_hybrid: HybridModel,
) -> None:
    """AC2: predict_goals returns home lambda >= 0 (clipped by model)."""
    h, _ = fitted_hybrid.predict_goals(_SAMPLE_FEATS)
    assert h >= 0.0


def test_when_hybrid_predicts_goals_then_away_goals_are_non_negative(
    fitted_hybrid: HybridModel,
) -> None:
    """AC2: predict_goals returns away lambda >= 0 (clipped by model)."""
    _, a = fitted_hybrid.predict_goals(_SAMPLE_FEATS)
    assert a >= 0.0


def test_when_hybrid_predicts_then_probabilities_sum_to_one(
    fitted_hybrid: HybridModel,
) -> None:
    """AC2: prob_home + prob_draw + prob_away == 1 (within floating-point tolerance)."""
    pred = fitted_hybrid.predict(_SAMPLE_FEATS)
    total = pred.prob_home + pred.prob_draw + pred.prob_away
    assert abs(total - 1.0) < _SIMPLEX_TOL


def test_when_hybrid_predicts_then_each_probability_is_in_unit_interval(
    fitted_hybrid: HybridModel,
) -> None:
    """AC2: each of prob_home, prob_draw, prob_away is in [0, 1]."""
    pred = fitted_hybrid.predict(_SAMPLE_FEATS)
    assert 0.0 <= pred.prob_home <= 1.0
    assert 0.0 <= pred.prob_draw <= 1.0
    assert 0.0 <= pred.prob_away <= 1.0


# ---------------------------------------------------------------------------
# AC3 — Ordered-logit output: W/D/L probs sum to 1
# ---------------------------------------------------------------------------


def test_when_ordered_logit_predicts_then_probabilities_sum_to_one(
    fitted_logit: OrderedLogitModel,
) -> None:
    """AC3: every OutcomeProbabilities row sums to 1 within tolerance."""
    X = pd.DataFrame({"elo_diff": [-200.0, -100.0, 0.0, 100.0, 200.0]})
    for row in fitted_logit.predict(X):
        total = row.prob_home + row.prob_draw + row.prob_away
        assert abs(total - 1.0) < _SIMPLEX_TOL, f"probs sum to {total}"


def test_when_ordered_logit_predicts_then_each_probability_is_in_unit_interval(
    fitted_logit: OrderedLogitModel,
) -> None:
    """AC3: each probability in the OutcomeProbabilities result is in [0, 1]."""
    row = fitted_logit.predict(pd.DataFrame({"elo_diff": [0.0]}))[0]
    assert 0.0 <= row.prob_home <= 1.0
    assert 0.0 <= row.prob_draw <= 1.0
    assert 0.0 <= row.prob_away <= 1.0


# ---------------------------------------------------------------------------
# AC4 — Determinism: same seed → identical predictions for both models
# ---------------------------------------------------------------------------


def test_when_hybrid_is_refit_with_same_seed_then_goal_predictions_are_identical(
    pipeline_dataset: MatchDataset,
) -> None:
    """AC4: two HybridModels with identical random_seed produce identical goal predictions."""
    m1 = HybridModel(random_seed=7).fit(pipeline_dataset)
    m2 = HybridModel(random_seed=7).fit(pipeline_dataset)
    h1, a1 = m1.predict_goals(_SAMPLE_FEATS)
    h2, a2 = m2.predict_goals(_SAMPLE_FEATS)
    assert np.isclose(h1, h2) and np.isclose(a1, a2)


def test_when_hybrid_is_refit_with_same_seed_then_outcome_probabilities_are_identical(
    pipeline_dataset: MatchDataset,
) -> None:
    """AC4: same seed → identical W/D/L distribution from the score matrix."""
    m1 = HybridModel(random_seed=7).fit(pipeline_dataset)
    m2 = HybridModel(random_seed=7).fit(pipeline_dataset)
    p1 = m1.predict(_SAMPLE_FEATS)
    p2 = m2.predict(_SAMPLE_FEATS)
    assert np.isclose(p1.prob_home, p2.prob_home)
    assert np.isclose(p1.prob_draw, p2.prob_draw)
    assert np.isclose(p1.prob_away, p2.prob_away)


def test_when_ordered_logit_is_refit_on_same_data_then_predictions_are_identical(
    pipeline_dataset: MatchDataset,
) -> None:
    """AC4: BFGS on a convex objective is deterministic — identical data → identical result."""
    m1 = OrderedLogitModel().fit(pipeline_dataset)
    m2 = OrderedLogitModel().fit(pipeline_dataset)
    X = pd.DataFrame({"elo_diff": [-50.0, 0.0, 50.0]})
    r1, r2 = m1.predict(X), m2.predict(X)
    for a, b in zip(r1, r2):
        assert abs(a.prob_home - b.prob_home) < 1e-10
        assert abs(a.prob_draw - b.prob_draw) < 1e-10
        assert abs(a.prob_away - b.prob_away) < 1e-10


# ---------------------------------------------------------------------------
# AC5 — ClassifierFactory.create builds all three legacy baselines
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["svm", "random_forest", "naive_bayes"])
def test_when_classifier_factory_creates_baseline_then_sklearn_protocol_is_satisfied(
    name: str,
) -> None:
    """AC5: ClassifierFactory.create still produces a non-None object with fit/predict."""
    clf = ClassifierFactory.create(name, TrainingConfig())
    assert clf is not None
    assert hasattr(clf, "fit"), f"classifier {name!r} missing .fit"
    assert hasattr(clf, "predict"), f"classifier {name!r} missing .predict"


# ---------------------------------------------------------------------------
# AC6 — worldcup_playoff.models re-exports full Cycle-4 + legacy surface
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "symbol",
    [
        # Legacy surface
        "ClassifierFactory",
        "ClassifierTrainer",
        "ModelEvaluator",
        # Cycle-4 hybrid
        "HybridModel",
        "HybridGoalModel",
        "GoalPrediction",
        "fit_hybrid",
        # Cycle-4 ordered logit
        "OrderedLogitModel",
        "OutcomeProbabilities",
        "fit_ordered_logit",
        # Cycle-4 dataset utility
        "MatchDataset",
        "build_dataset",
    ],
)
def test_when_models_package_is_imported_then_symbol_is_accessible(symbol: str) -> None:
    """AC6: every declared Cycle-4 and legacy symbol is accessible from worldcup_playoff.models."""
    import worldcup_playoff.models as pkg

    assert hasattr(pkg, symbol), f"{symbol!r} not found in worldcup_playoff.models"
