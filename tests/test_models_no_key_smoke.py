"""No-key smoke tests for Issue #15 (corrected from source-blind draft).

The R4 test-designer phase produced this file from the acceptance criteria
without reading the implementation.  The source-blind draft used incorrect API
assumptions (sklearn-style fit(X,y)/predict(X), lowercase column schema, wrong
constructor keyword).  This revision aligns the tests with the real API while
preserving the original behavioural intent of each criterion.

Criteria covered (oracle UNIT-verifiable):
  Criterion 1  — end-to-end no-key pipeline
  Criterion 4  — determinism: same seed → identical predictions
  Criterion 5  — ClassifierFactory.create builds svm/random_forest/naive_bayes
  Criterion 6  — worldcup_playoff.models re-exports Cycle-4 + legacy symbols

Property-based (Hypothesis @given) tests are kept for criterion 4 and 5.
Criteria the oracle marked NOT VERIFIABLE are still omitted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from worldcup_playoff.config import TrainingConfig
from worldcup_playoff.data.elo import compute_elo
from worldcup_playoff.features.build import build_features
from worldcup_playoff.models.dataset import MatchDataset, add_targets, build_dataset
from worldcup_playoff.models.hybrid import HybridModel
from worldcup_playoff.models.ordered_logit import OrderedLogitModel
from worldcup_playoff.simulation.poisson import fit_dixon_coles

# ---------------------------------------------------------------------------
# Shared fixture helpers (martj42 UPPERCASE schema — no network access)
# ---------------------------------------------------------------------------

_TEAMS = [
    "Brazil",
    "Argentina",
    "France",
    "Germany",
    "Spain",
    "England",
    "Italy",
    "Portugal",
]

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


def _make_martj42(n: int = 48) -> pd.DataFrame:
    """Minimal martj42 UPPERCASE-schema DataFrame; all matches have known scores."""
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


def _make_dataset(n: int = 48) -> MatchDataset:
    """Run the full no-key pipeline and return a MatchDataset."""
    df = _make_martj42(n)
    elo = compute_elo(df)
    abilities = fit_dixon_coles(df)
    features = add_targets(build_features(df, elo, abilities))
    return build_dataset(features, test_size=0.2, feature_cols=_FEATURE_COLS)


# Module-level singleton to avoid re-running the pipeline in Hypothesis tests
_SHARED_DATASET: MatchDataset | None = None


def _shared_dataset() -> MatchDataset:
    global _SHARED_DATASET
    if _SHARED_DATASET is None:
        _SHARED_DATASET = _make_dataset()
    return _SHARED_DATASET


# ---------------------------------------------------------------------------
# Criterion 1 — end-to-end no-key pipeline  [UNIT]
# ---------------------------------------------------------------------------


def test_when_martj42_fixture_is_piped_through_full_pipeline_then_hybrid_model_is_fitted():
    """
    Criterion 1: compute_elo → fit_dixon_coles → build_features → build_dataset →
    HybridModel.fit produces a model that returns a valid prediction.

    Corrected from source-blind draft: uses UPPERCASE martj42 schema, calls
    add_targets before build_dataset, uses HybridModel(random_seed=…).fit(dataset).
    """
    ds = _make_dataset()
    model = HybridModel(random_seed=0).fit(ds)
    h, a = model.predict_goals(_SAMPLE_FEATS)
    assert isinstance(h, float) and isinstance(a, float)


def test_when_martj42_fixture_is_piped_through_full_pipeline_then_ordered_logit_model_is_fitted():
    """
    Criterion 1 (continued): the same pipeline feeds OrderedLogitModel successfully.

    Corrected from source-blind draft: uses OrderedLogitModel().fit(dataset)
    and predict(DataFrame) instead of predict(array).
    """
    ds = _make_dataset()
    model = OrderedLogitModel().fit(ds)
    result = model.predict(pd.DataFrame({"elo_diff": [0.0]}))
    assert len(result) == 1


# ---------------------------------------------------------------------------
# Criterion 4 — determinism  [UNIT + property]
# ---------------------------------------------------------------------------


def test_when_hybrid_model_is_refit_with_same_seed_then_predictions_are_identical():
    """Criterion 4: identical random_seed → identical HybridModel.predict_goals."""
    ds = _shared_dataset()
    h1, a1 = HybridModel(random_seed=42).fit(ds).predict_goals(_SAMPLE_FEATS)
    h2, a2 = HybridModel(random_seed=42).fit(ds).predict_goals(_SAMPLE_FEATS)
    assert np.isclose(h1, h2) and np.isclose(a1, a2)


def test_when_ordered_logit_model_is_refit_with_same_data_then_predictions_are_identical():
    """Criterion 4: BFGS on a convex objective is deterministic — same data → same output."""
    ds = _shared_dataset()
    X = pd.DataFrame({"elo_diff": [-50.0, 0.0, 50.0]})
    r1 = OrderedLogitModel().fit(ds).predict(X)
    r2 = OrderedLogitModel().fit(ds).predict(X)
    for a, b in zip(r1, r2):
        assert abs(a.prob_home - b.prob_home) < 1e-10
        assert abs(a.prob_draw - b.prob_draw) < 1e-10
        assert abs(a.prob_away - b.prob_away) < 1e-10


@given(st.integers(min_value=0, max_value=9999))
@settings(max_examples=5, deadline=None)
def test_when_hybrid_is_refit_with_any_seed_then_output_is_reproducible(seed: int) -> None:
    """
    Criterion 4 (property): for every seed value, fitting HybridModel twice with
    that same seed always yields identical goal predictions — determinism is an
    unconditional invariant over the seed domain.
    """
    ds = _shared_dataset()
    h1, a1 = HybridModel(random_seed=seed).fit(ds).predict_goals(_SAMPLE_FEATS)
    h2, a2 = HybridModel(random_seed=seed).fit(ds).predict_goals(_SAMPLE_FEATS)
    assert np.isclose(h1, h2) and np.isclose(a1, a2)


# ---------------------------------------------------------------------------
# Criterion 5 — ClassifierFactory baseline coexistence  [UNIT + property]
# ---------------------------------------------------------------------------


def test_when_classifier_factory_creates_svm_then_a_classifier_is_returned():
    """Criterion 5: ClassifierFactory.create('svm', config) returns a non-None baseline."""
    from worldcup_playoff.models.classifiers import ClassifierFactory

    assert ClassifierFactory.create("svm", TrainingConfig()) is not None


def test_when_classifier_factory_creates_random_forest_then_a_classifier_is_returned():
    """Criterion 5: ClassifierFactory.create('random_forest', config) returns a baseline."""
    from worldcup_playoff.models.classifiers import ClassifierFactory

    assert ClassifierFactory.create("random_forest", TrainingConfig()) is not None


def test_when_classifier_factory_creates_naive_bayes_then_a_classifier_is_returned():
    """Criterion 5: ClassifierFactory.create('naive_bayes', config) returns a baseline."""
    from worldcup_playoff.models.classifiers import ClassifierFactory

    assert ClassifierFactory.create("naive_bayes", TrainingConfig()) is not None


@given(st.sampled_from(["svm", "random_forest", "naive_bayes"]))
def test_when_any_baseline_name_is_passed_to_factory_then_sklearn_protocol_is_satisfied(
    name: str,
) -> None:
    """
    Criterion 5 (property): for every valid baseline name, ClassifierFactory.create
    returns an object with .fit and .predict — the sklearn Protocol is satisfied for
    all three names unconditionally.
    """
    from worldcup_playoff.models.classifiers import ClassifierFactory

    clf = ClassifierFactory.create(name, TrainingConfig())
    assert hasattr(clf, "fit")
    assert hasattr(clf, "predict")


# ---------------------------------------------------------------------------
# Criterion 6 — worldcup_playoff.models re-exports  [UNIT + property]
# ---------------------------------------------------------------------------


def test_when_models_package_is_imported_then_legacy_exports_are_accessible():
    """Criterion 6: ClassifierFactory, ClassifierTrainer, ModelEvaluator still exported."""
    import worldcup_playoff.models as pkg

    assert hasattr(pkg, "ClassifierFactory")
    assert hasattr(pkg, "ClassifierTrainer")
    assert hasattr(pkg, "ModelEvaluator")


def test_when_models_package_is_imported_then_cycle4_hybrid_model_is_accessible():
    """Criterion 6: HybridModel is re-exported from worldcup_playoff.models."""
    import worldcup_playoff.models as pkg

    assert hasattr(pkg, "HybridModel")


def test_when_models_package_is_imported_then_cycle4_ordered_logit_model_is_accessible():
    """Criterion 6: OrderedLogitModel is re-exported from worldcup_playoff.models."""
    import worldcup_playoff.models as pkg

    assert hasattr(pkg, "OrderedLogitModel")


def test_when_models_package_is_imported_then_cycle4_dataset_builder_is_accessible():
    """Criterion 6: build_dataset is re-exported from worldcup_playoff.models."""
    import worldcup_playoff.models as pkg

    assert hasattr(pkg, "build_dataset")


@given(
    st.sampled_from(
        [
            "ClassifierFactory",
            "ClassifierTrainer",
            "ModelEvaluator",
            "HybridModel",
            "OrderedLogitModel",
            "build_dataset",
        ]
    )
)
def test_when_any_expected_name_is_looked_up_in_models_package_then_it_is_present(
    name: str,
) -> None:
    """
    Criterion 6 (property): every name in the declared Cycle-4 + legacy surface is
    unconditionally present in worldcup_playoff.models — no name may be missing.
    """
    import worldcup_playoff.models as pkg

    assert hasattr(pkg, name)
