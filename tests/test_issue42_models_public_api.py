"""Source-blind example tests for issue #42.

Issue: feat: models package public API + baseline retention + no-key integration
       smoke test (models/__init__.py)

Derived exclusively from the acceptance criteria text and
.code-generator/requirements.md.  No implementation source was opened during
authoring — tests are written in the Red phase of TDD.

Verifiable acceptance criteria covered (per oracle report):
  AC1  models/__init__.py re-exports MatchDataset/build_dataset,
       HybridModel/GoalPrediction/fit_hybrid,
       OrderedLogitModel/OutcomeProbabilities/fit_ordered_logit, and the
       retained baseline ClassifierFactory/ClassifierTrainer/ModelEvaluator,
       with a stable __all__.                                              [UNIT]
  AC2  Legacy classifiers remain importable and functional as a baseline
       (no new responsibilities added to them).                            [UNIT]
  AC3  No-key, no-network integration smoke test that fits all three tiers
       from a synthetic feature frame: hybrid goal+margin output,
       ordered-logit W/D/L triple summing to ~1.0, baseline classifier
       prediction.                                                         [UNIT]

Skipped (oracle: NOT VERIFIABLE):
  AC5  SOLID / clean-code prose — subjective.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st


# ---------------------------------------------------------------------------
# Official confederation tuple (requirements.md §Data Contracts)
# ---------------------------------------------------------------------------
_CONFEDERATIONS = ("UEFA", "CONMEBOL", "CAF", "AFC", "CONCACAF", "OFC")

# Complete set of names the public API must expose.
_REQUIRED_NAMES: frozenset[str] = frozenset(
    {
        "MatchDataset",
        "build_dataset",
        "HybridModel",
        "GoalPrediction",
        "fit_hybrid",
        "OrderedLogitModel",
        "OutcomeProbabilities",
        "fit_ordered_logit",
        "ClassifierFactory",
        "ClassifierTrainer",
        "ModelEvaluator",
    }
)


# ---------------------------------------------------------------------------
# Synthetic-data helpers — built from the covariate spec in requirements.md;
# no production source was consulted.
# ---------------------------------------------------------------------------


def _hybrid_match_df(n: int = 60, seed: int = 0) -> pd.DataFrame:
    """
    Synthetic match frame matching the Groll-hybrid feature spec
    (requirements.md §Groll RF hybrid): Elo, Elo-diff, Dixon-Coles
    attack/defence, rest days, confederation, neutral flag, plus
    target columns home_goals / away_goals.
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
            "confederation": rng.choice(list(_CONFEDERATIONS), n),
            "neutral": rng.choice([True, False], n),
            "home_goals": rng.integers(0, 5, n).astype(float),
            "away_goals": rng.integers(0, 5, n).astype(float),
        }
    )


def _logit_df(n: int = 80, seed: int = 0) -> pd.DataFrame:
    """
    Synthetic frame for the Elo-diff ordered-logit model.
    Outcome encoding: away=0, draw=1, home=2.
    All three classes are always represented (statsmodels requirement).
    """
    rng = np.random.default_rng(seed)
    elo_diff = rng.uniform(-350.0, 350.0, size=n)
    y_outcome = np.where(elo_diff > 100, 2, np.where(elo_diff < -100, 0, 1))
    df = pd.DataFrame({"elo_diff": elo_diff, "y_outcome": y_outcome.astype(int)})
    for cls in (0, 1, 2):
        if cls not in df["y_outcome"].values:
            df = pd.concat(
                [df, pd.DataFrame({"elo_diff": [float((cls - 1) * 150)], "y_outcome": [cls]})],
                ignore_index=True,
            )
    return df


def _feature_row() -> pd.DataFrame:
    """Single-row inference frame — target columns removed."""
    row = _hybrid_match_df(n=1, seed=42)
    return row.drop(columns=["home_goals", "away_goals"]).reset_index(drop=True)


class _MinimalDataset:
    """Duck-typed MatchDataset stub (criterion requires only .train to exist)."""

    def __init__(self, train: pd.DataFrame) -> None:
        self.train = train


# ===========================================================================
# AC1 — Public API: individual re-export checks
# ===========================================================================


def test_when_models_package_imported_then_MatchDataset_is_accessible():
    from worldcup_playoff.models import MatchDataset  # noqa: F401

    assert MatchDataset is not None


def test_when_models_package_imported_then_build_dataset_is_accessible():
    from worldcup_playoff.models import build_dataset

    assert callable(build_dataset)


def test_when_models_package_imported_then_HybridModel_is_accessible():
    from worldcup_playoff.models import HybridModel  # noqa: F401

    assert HybridModel is not None


def test_when_models_package_imported_then_GoalPrediction_is_accessible():
    from worldcup_playoff.models import GoalPrediction  # noqa: F401

    assert GoalPrediction is not None


def test_when_models_package_imported_then_fit_hybrid_is_accessible():
    from worldcup_playoff.models import fit_hybrid

    assert callable(fit_hybrid)


def test_when_models_package_imported_then_OrderedLogitModel_is_accessible():
    from worldcup_playoff.models import OrderedLogitModel  # noqa: F401

    assert OrderedLogitModel is not None


def test_when_models_package_imported_then_OutcomeProbabilities_is_accessible():
    from worldcup_playoff.models import OutcomeProbabilities  # noqa: F401

    assert OutcomeProbabilities is not None


def test_when_models_package_imported_then_fit_ordered_logit_is_accessible():
    from worldcup_playoff.models import fit_ordered_logit

    assert callable(fit_ordered_logit)


def test_when_models_package_imported_then_ClassifierFactory_is_accessible():
    from worldcup_playoff.models import ClassifierFactory  # noqa: F401

    assert ClassifierFactory is not None


def test_when_models_package_imported_then_ClassifierTrainer_is_accessible():
    from worldcup_playoff.models import ClassifierTrainer  # noqa: F401

    assert ClassifierTrainer is not None


def test_when_models_package_imported_then_ModelEvaluator_is_accessible():
    from worldcup_playoff.models import ModelEvaluator  # noqa: F401

    assert ModelEvaluator is not None


# --- __all__ stability ---


def test_when_models_all_inspected_then_it_exists():
    """models package must declare __all__."""
    import worldcup_playoff.models as m

    assert hasattr(m, "__all__"), "worldcup_playoff.models must define __all__"


def test_when_models_all_inspected_then_all_required_names_are_present():
    """Every name listed in the acceptance criteria must appear in __all__."""
    import worldcup_playoff.models as m

    missing = _REQUIRED_NAMES - set(m.__all__)
    assert not missing, f"Names absent from __all__: {sorted(missing)}"


def test_when_models_all_inspected_then_every_declared_name_is_accessible():
    """
    Stability invariant: no name may appear in __all__ without being
    reachable as an attribute.  Catches stale entries.
    """
    import worldcup_playoff.models as m

    broken = [name for name in m.__all__ if not hasattr(m, name)]
    assert not broken, f"Names in __all__ but not accessible: {broken}"


# Property: all-names-accessible is an ordering/completeness invariant
# over every element of __all__, not just the required subset.
@given(st.data())
@settings(max_examples=30)
def test_when_name_sampled_from_all_then_it_resolves_to_a_module_attribute(
    data: st.DataObject,
) -> None:
    """
    Structural ordering invariant (AC1): for any name drawn from __all__,
    getattr(module, name) must not raise AttributeError.
    """
    import worldcup_playoff.models as m

    if not m.__all__:
        return
    name = data.draw(st.sampled_from(list(m.__all__)))
    assert hasattr(m, name), f"'{name}' is in __all__ but not an attribute"


# ===========================================================================
# AC2 — Legacy classifiers remain importable and functional as baseline
# ===========================================================================


def test_when_ClassifierFactory_imported_from_models_then_it_has_create_method():
    from worldcup_playoff.models import ClassifierFactory

    assert callable(getattr(ClassifierFactory, "create", None)), (
        "ClassifierFactory must expose a callable 'create' method"
    )


def test_when_ClassifierTrainer_imported_from_models_then_it_has_train_method():
    from worldcup_playoff.models import ClassifierTrainer

    assert callable(getattr(ClassifierTrainer, "train", None)), (
        "ClassifierTrainer must expose a callable 'train' method"
    )


def test_when_ModelEvaluator_imported_from_models_then_it_is_instantiable():
    from worldcup_playoff.models import ModelEvaluator

    assert ModelEvaluator is not None
    assert isinstance(ModelEvaluator, type) or callable(ModelEvaluator), (
        "ModelEvaluator must be a class or callable"
    )


def test_when_ClassifierFactory_creates_svm_then_sklearn_protocol_is_satisfied():
    """
    Legacy classifiers follow the Classifier Protocol (requirements.md):
    any object with fit(X, y) and predict(X) qualifies.  Creating an SVM
    via ClassifierFactory must produce such an object.

    ClassifierFactory.create is a static method: create(name, config).
    The training config lives at AppConfig().training (TrainingConfig).
    """
    from worldcup_playoff.models import ClassifierFactory
    from worldcup_playoff.config import AppConfig

    clf = ClassifierFactory.create("svm", AppConfig().training)
    assert callable(getattr(clf, "fit", None)), (
        "ClassifierFactory.create('svm') result must have .fit()"
    )
    assert callable(getattr(clf, "predict", None)), (
        "ClassifierFactory.create('svm') result must have .predict()"
    )


def test_when_ClassifierFactory_creates_random_forest_then_sklearn_protocol_is_satisfied():
    """Same Protocol check for the random_forest variant."""
    from worldcup_playoff.models import ClassifierFactory
    from worldcup_playoff.config import AppConfig

    clf = ClassifierFactory.create("random_forest", AppConfig().training)
    assert callable(getattr(clf, "fit", None))
    assert callable(getattr(clf, "predict", None))


def test_when_ClassifierFactory_creates_naive_bayes_then_sklearn_protocol_is_satisfied():
    """Same Protocol check for the naive_bayes variant."""
    from worldcup_playoff.models import ClassifierFactory
    from worldcup_playoff.config import AppConfig

    clf = ClassifierFactory.create("naive_bayes", AppConfig().training)
    assert callable(getattr(clf, "fit", None))
    assert callable(getattr(clf, "predict", None))


def test_when_svm_classifier_fitted_then_prediction_is_binary():
    """
    Legacy classifiers predict HOME_WIN (0/1).  Manually fitting the
    Protocol-compliant object must return values in {0, 1}.
    This is the 'functional as a baseline' check — no new responsibilities,
    same contract as before.
    """
    from worldcup_playoff.models import ClassifierFactory
    from worldcup_playoff.config import AppConfig

    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (24, 10))
    y = np.array([0, 1] * 12)

    clf = ClassifierFactory.create("svm", AppConfig().training)
    clf.fit(X, y)
    predictions = clf.predict(X[:4])
    assert all(int(p) in (0, 1) for p in predictions), (
        f"Legacy classifier must predict 0 or 1; got {predictions}"
    )


# ===========================================================================
# AC3 — No-key, no-network integration smoke test (all three tiers)
# ===========================================================================


def test_when_hybrid_model_fitted_on_synthetic_frame_then_GoalPrediction_is_returned():
    """AC3: HybridModel.predict returns a GoalPrediction instance."""
    from worldcup_playoff.models import GoalPrediction, MatchDataset, fit_hybrid

    df = _hybrid_match_df(n=60, seed=1)
    dataset = MatchDataset(train=df.head(45), test=df.tail(15))
    model = fit_hybrid(dataset)
    result = model.predict(_feature_row())
    assert isinstance(result, GoalPrediction)


def test_when_hybrid_model_predicts_then_home_goals_are_non_negative():
    """AC3: GoalPrediction.home_goals >= 0 (non-negative clamp)."""
    from worldcup_playoff.models import MatchDataset, fit_hybrid

    df = _hybrid_match_df(n=60, seed=2)
    model = fit_hybrid(MatchDataset(train=df.head(45), test=df.tail(15)))
    result = model.predict(_feature_row())
    assert result.home_goals >= 0.0


def test_when_hybrid_model_predicts_then_away_goals_are_non_negative():
    """AC3: GoalPrediction.away_goals >= 0 (non-negative clamp)."""
    from worldcup_playoff.models import MatchDataset, fit_hybrid

    df = _hybrid_match_df(n=60, seed=3)
    model = fit_hybrid(MatchDataset(train=df.head(45), test=df.tail(15)))
    result = model.predict(_feature_row())
    assert result.away_goals >= 0.0


def test_when_hybrid_model_predicts_then_margin_equals_home_minus_away_goals():
    """AC3: GoalPrediction.margin = home_goals − away_goals."""
    from worldcup_playoff.models import MatchDataset, fit_hybrid

    df = _hybrid_match_df(n=60, seed=4)
    model = fit_hybrid(MatchDataset(train=df.head(45), test=df.tail(15)))
    result = model.predict(_feature_row())
    assert result.margin == pytest.approx(result.home_goals - result.away_goals, abs=1e-9)


def test_when_hybrid_model_predicts_then_wdl_probs_sum_to_one():
    """AC3: W/D/L probability triple from the score matrix sums to 1.0."""
    from worldcup_playoff.models import MatchDataset, fit_hybrid

    df = _hybrid_match_df(n=60, seed=5)
    model = fit_hybrid(MatchDataset(train=df.head(45), test=df.tail(15)))
    result = model.predict(_feature_row())
    total = result.prob_home + result.prob_draw + result.prob_away
    assert total == pytest.approx(1.0, abs=1e-6)


def test_when_ordered_logit_fitted_on_synthetic_frame_then_OutcomeProbabilities_is_returned():
    """AC3: OrderedLogitModel.predict returns OutcomeProbabilities objects."""
    from worldcup_playoff.models import OutcomeProbabilities, fit_ordered_logit

    model = fit_ordered_logit(_MinimalDataset(_logit_df(n=80, seed=0)), config=None)  # type: ignore[arg-type]
    results = model.predict(pd.DataFrame({"elo_diff": [100.0, -50.0, 0.0]}))
    assert isinstance(results, list)
    assert len(results) == 3
    assert all(isinstance(r, OutcomeProbabilities) for r in results)


def test_when_ordered_logit_predicts_then_wdl_triple_sums_to_approximately_one():
    """AC3: ordered-logit W/D/L sum to ~1.0 for every output row."""
    from worldcup_playoff.models import fit_ordered_logit

    model = fit_ordered_logit(_MinimalDataset(_logit_df(n=80, seed=0)), config=None)  # type: ignore[arg-type]
    for op in model.predict(pd.DataFrame({"elo_diff": [200.0, 0.0, -200.0]})):
        total = op.prob_home + op.prob_draw + op.prob_away
        assert total == pytest.approx(1.0, abs=1e-5), (
            f"W/D/L must sum to 1.0; got {total} "
            f"(home={op.prob_home}, draw={op.prob_draw}, away={op.prob_away})"
        )


def test_when_ordered_logit_predicts_then_each_probability_is_in_unit_interval():
    """AC3: every individual W/D/L probability lies in [0, 1]."""
    from worldcup_playoff.models import fit_ordered_logit

    model = fit_ordered_logit(_MinimalDataset(_logit_df(n=80, seed=0)), config=None)  # type: ignore[arg-type]
    for op in model.predict(pd.DataFrame({"elo_diff": [100.0]})):
        assert 0.0 <= op.prob_home <= 1.0
        assert 0.0 <= op.prob_draw <= 1.0
        assert 0.0 <= op.prob_away <= 1.0


def test_when_baseline_classifier_fitted_on_synthetic_frame_then_prediction_is_binary():
    """AC3: the retained legacy classifier produces a binary (0/1) prediction."""
    from worldcup_playoff.models import ClassifierFactory
    from worldcup_playoff.config import AppConfig

    rng = np.random.default_rng(42)
    X_train = rng.uniform(0.0, 1.0, (30, 10))
    y_train = np.array([0, 1] * 15)

    clf = ClassifierFactory.create("svm", AppConfig().training)
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_train[:1])
    assert int(prediction[0]) in (0, 1), (
        f"Legacy baseline classifier must return 0 or 1; got {prediction[0]}"
    )


def test_when_all_three_tiers_executed_then_no_network_call_is_required():
    """
    AC3: the entire three-tier smoke test must be self-contained
    (no API key, no HTTP, no file-system writes).  Completing without
    ImportError or ConnectionError is the observable assertion.
    """
    from worldcup_playoff.models import (
        MatchDataset,
        fit_hybrid,
        GoalPrediction,
        OutcomeProbabilities,
        fit_ordered_logit,
        ClassifierFactory,
    )
    from worldcup_playoff.config import AppConfig

    # Tier 1 — Groll RF hybrid
    df = _hybrid_match_df(n=60, seed=99)
    hybrid = fit_hybrid(MatchDataset(train=df.head(45), test=df.tail(15)))
    gp = hybrid.predict(_feature_row())
    assert isinstance(gp, GoalPrediction)
    assert gp.home_goals >= 0.0
    assert gp.away_goals >= 0.0
    assert gp.margin == pytest.approx(gp.home_goals - gp.away_goals, abs=1e-9)

    # Tier 2 — Elo-diff ordered logit
    ol = fit_ordered_logit(_MinimalDataset(_logit_df(n=80, seed=99)), config=None)  # type: ignore[arg-type]
    ops = ol.predict(pd.DataFrame({"elo_diff": [0.0]}))
    op = ops[0]
    assert isinstance(op, OutcomeProbabilities)
    assert op.prob_home + op.prob_draw + op.prob_away == pytest.approx(1.0, abs=1e-5)

    # Tier 3 — legacy baseline classifier
    rng = np.random.default_rng(99)
    X = rng.uniform(0.0, 1.0, (30, 10))
    y = np.array([0, 1] * 15)
    clf = ClassifierFactory.create("svm", AppConfig().training)
    clf.fit(X, y)
    pred = clf.predict(X[:1])
    assert int(pred[0]) in (0, 1)


# ===========================================================================
# Property-based tests — invariants derived from the acceptance criteria
# ===========================================================================


@given(
    features=st.fixed_dictionaries(
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
            "confederation": st.sampled_from(_CONFEDERATIONS),
            "neutral": st.booleans(),
        }
    )
)
@settings(max_examples=25, deadline=None)
def test_when_any_valid_hybrid_feature_row_given_then_margin_equals_home_minus_away(
    features: dict,
) -> None:
    """
    Ordering invariant (AC3): margin = home_goals − away_goals for ALL valid
    feature inputs, not just the fixed example above.
    """

    # Re-use a cached trained model to avoid re-fitting inside Hypothesis.
    # The model is module-level so Hypothesis sees the same object every call.
    result = _SHARED_HYBRID.predict(pd.DataFrame({k: [v] for k, v in features.items()}))  # type: ignore[arg-type]
    assert result.margin == pytest.approx(result.home_goals - result.away_goals, abs=1e-9)


@given(
    elo_diffs=st.lists(
        st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=10,
    )
)
@settings(max_examples=20, deadline=None)
def test_when_any_valid_elo_diff_batch_given_then_wdl_probs_sum_to_one(
    elo_diffs: list[float],
) -> None:
    """
    Ordering invariant (AC3): for any finite elo_diff batch, every
    OutcomeProbabilities row has all three components in [0, 1] and
    they sum to 1.0 — the invariant must hold across the full input domain.
    """
    ops = _SHARED_LOGIT.predict(pd.DataFrame({"elo_diff": elo_diffs}))  # type: ignore[arg-type]
    for op in ops:
        assert 0.0 <= op.prob_home <= 1.0
        assert 0.0 <= op.prob_draw <= 1.0
        assert 0.0 <= op.prob_away <= 1.0
        total = op.prob_home + op.prob_draw + op.prob_away
        assert total == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Module-level shared models — fitted once to amortise cost across the
# property-based tests above.  Built at collection time; no network needed.
# ---------------------------------------------------------------------------


def _build_shared_hybrid():
    try:
        from worldcup_playoff.models import MatchDataset, fit_hybrid

        df = _hybrid_match_df(n=60, seed=0)
        return fit_hybrid(MatchDataset(train=df.head(45), test=df.tail(15)))
    except Exception:  # noqa: BLE001 — models package not yet implemented (Red phase)
        return None


def _build_shared_logit():
    try:
        from worldcup_playoff.models import fit_ordered_logit

        return fit_ordered_logit(_MinimalDataset(_logit_df(n=80, seed=0)), config=None)  # type: ignore[arg-type]
    except Exception:  # noqa: BLE001
        return None


_SHARED_HYBRID = _build_shared_hybrid()
_SHARED_LOGIT = _build_shared_logit()
