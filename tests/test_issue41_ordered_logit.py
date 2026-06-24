"""
Tests for issue #41 — Elo-diff ordered-logit secondary/fallback model.

Criteria covered:
  AC1: Module exposes OrderedLogitModel (fit/predict), frozen OutcomeProbabilities, fit_ordered_logit.
  AC2: Fits on dataset.train only; deterministic (bfgs, fixed maxiter, disp=False).
  AC3: statsmodels is lazy-imported inside fit — not loaded at module-collection time.
  AC5: Probability rows map statsmodels [away, draw, home] order → prob_away, prob_draw, prob_home.

Skipped (oracle: not runtime-verifiable):
  AC4: "W/D/L only, no goal output, runs without API key".
  AC6: "All tests pass" — meta-criterion.
  AC7: SOLID / line-length prose — subjective.
"""

import sys

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st


# ---------------------------------------------------------------------------
# Fixtures / helpers — built entirely from criteria text, not from source
# ---------------------------------------------------------------------------


def _make_train_df(n: int = 80, seed: int = 0) -> pd.DataFrame:
    """
    Synthetic match records with an elo_diff feature and an ordinal outcome.

    Outcome encoding per issue spec (away=0, draw=1, home=2):
      0 = away win  |  1 = draw  |  2 = home win

    elo_diff > 0 means the home side has a higher Elo rating.
    Three classes are always represented (ordered logit requires all categories).
    """
    rng = np.random.default_rng(seed)
    elo_diff = rng.uniform(-350.0, 350.0, size=n)
    # home win when elo_diff > 100, away win when < -100, draw otherwise
    y_outcome = np.where(elo_diff > 100, 2, np.where(elo_diff < -100, 0, 1))
    df = pd.DataFrame({"elo_diff": elo_diff, "y_outcome": y_outcome.astype(int)})
    # Guarantee every class is present — ordered logit silently misbehaves otherwise
    for cls in (0, 1, 2):
        if cls not in df["y_outcome"].values:
            df = pd.concat(
                [df, pd.DataFrame({"elo_diff": [float((cls - 1) * 150)], "y_outcome": [cls]})],
                ignore_index=True,
            )
    return df


class _MinimalDataset:
    """Duck-typed MatchDataset stub — criterion only requires .train to exist."""

    def __init__(self, train: pd.DataFrame):
        self.train = train


class _DatasetWithTestSplit:
    """MatchDataset stub that also carries a .test attribute."""

    def __init__(self, train: pd.DataFrame, test: pd.DataFrame):
        self.train = train
        self.test = test


# ---------------------------------------------------------------------------
# AC1 — Public API contract
# ---------------------------------------------------------------------------


class TestPublicAPI:
    """Acceptance criterion: module exposes OrderedLogitModel, OutcomeProbabilities, factory."""

    def test_when_module_imported_then_OrderedLogitModel_is_accessible(self):
        from worldcup_playoff.models.ordered_logit import OrderedLogitModel  # noqa: F401

        assert callable(getattr(OrderedLogitModel, "fit", None)), (
            "OrderedLogitModel must have a callable 'fit' method"
        )
        assert callable(getattr(OrderedLogitModel, "predict", None)), (
            "OrderedLogitModel must have a callable 'predict' method"
        )

    def test_when_OutcomeProbabilities_constructed_then_fields_are_accessible(self):
        from worldcup_playoff.models.ordered_logit import OutcomeProbabilities

        op = OutcomeProbabilities(prob_home=0.5, prob_draw=0.3, prob_away=0.2)
        assert hasattr(op, "prob_home")
        assert hasattr(op, "prob_draw")
        assert hasattr(op, "prob_away")

    def test_when_OutcomeProbabilities_field_assigned_then_AttributeError_is_raised(self):
        # Criterion: OutcomeProbabilities is *frozen* — mutation must be rejected.
        from worldcup_playoff.models.ordered_logit import OutcomeProbabilities

        op = OutcomeProbabilities(prob_home=0.5, prob_draw=0.3, prob_away=0.2)
        with pytest.raises((AttributeError, TypeError)):
            op.prob_home = 0.99  # type: ignore[misc]

    def test_when_module_imported_then_fit_ordered_logit_factory_is_callable(self):
        from worldcup_playoff.models.ordered_logit import fit_ordered_logit  # noqa: F401

        assert callable(fit_ordered_logit)

    def test_when_factory_called_with_dataset_and_config_then_OrderedLogitModel_is_returned(self):
        # Criterion: fit_ordered_logit(dataset, config) is a factory that produces the model.
        # Passes None so the factory uses its default OrderedLogitConfig (maxiter=100, features=["elo_diff"]).
        from worldcup_playoff.models.ordered_logit import fit_ordered_logit, OrderedLogitModel

        dataset = _MinimalDataset(_make_train_df())
        model = fit_ordered_logit(dataset, config=None)  # type: ignore[arg-type]
        assert isinstance(model, OrderedLogitModel)

    def test_when_predict_called_after_fit_then_list_of_OutcomeProbabilities_is_returned(self):
        from worldcup_playoff.models.ordered_logit import OrderedLogitModel, OutcomeProbabilities

        model = OrderedLogitModel()
        model.fit(_MinimalDataset(_make_train_df()))  # type: ignore[arg-type]
        result = model.predict(pd.DataFrame({"elo_diff": [100.0, -50.0, 0.0]}))

        assert isinstance(result, list), "predict must return a list"
        assert len(result) == 3, "result length must match number of input rows"
        assert all(isinstance(r, OutcomeProbabilities) for r in result)


# ---------------------------------------------------------------------------
# AC2 — Fits on dataset.train only; deterministic
# ---------------------------------------------------------------------------


class TestFitBehaviour:
    """Acceptance criterion: fit uses only .train; bfgs + fixed maxiter → deterministic."""

    def test_when_fit_called_twice_with_same_data_then_predictions_are_identical(self):
        from worldcup_playoff.models.ordered_logit import OrderedLogitModel

        df = _make_train_df(seed=42)
        m1, m2 = OrderedLogitModel(), OrderedLogitModel()
        m1.fit(_MinimalDataset(df))  # type: ignore[arg-type]
        m2.fit(_MinimalDataset(df))  # type: ignore[arg-type]

        probe = pd.DataFrame({"elo_diff": [50.0, -200.0, 0.0]})
        for a, b in zip(m1.predict(probe), m2.predict(probe)):
            assert a.prob_home == pytest.approx(b.prob_home, abs=1e-9)
            assert a.prob_draw == pytest.approx(b.prob_draw, abs=1e-9)
            assert a.prob_away == pytest.approx(b.prob_away, abs=1e-9)

    def test_when_dataset_test_split_differs_then_predictions_are_unchanged(self):
        """
        Criterion: "Fits on dataset.train only — no internal split/shuffle."
        Two datasets with identical .train but different .test must yield identical predictions.
        """
        from worldcup_playoff.models.ordered_logit import OrderedLogitModel

        train_df = _make_train_df(seed=7)
        other_test_df = _make_train_df(n=20, seed=99)

        ds_with_test = _DatasetWithTestSplit(train_df, other_test_df)
        ds_no_test = _MinimalDataset(train_df)

        m_with = OrderedLogitModel()
        m_without = OrderedLogitModel()
        m_with.fit(ds_with_test)  # type: ignore[arg-type]
        m_without.fit(ds_no_test)  # type: ignore[arg-type]

        probe = pd.DataFrame({"elo_diff": [75.0, -75.0]})
        for a, b in zip(m_with.predict(probe), m_without.predict(probe)):
            assert a.prob_home == pytest.approx(b.prob_home, abs=1e-9)
            assert a.prob_draw == pytest.approx(b.prob_draw, abs=1e-9)
            assert a.prob_away == pytest.approx(b.prob_away, abs=1e-9)


# ---------------------------------------------------------------------------
# AC3 — statsmodels is lazy-imported (not pulled in at collection time)
# ---------------------------------------------------------------------------


class TestLazyImport:
    """Acceptance criterion: module import must NOT trigger a statsmodels import."""

    def test_when_module_imported_then_statsmodels_is_not_yet_in_sys_modules(self):
        """
        Force-evict statsmodels from sys.modules, re-import the ordered_logit module,
        and verify that statsmodels is still absent — confirming the import is deferred.
        """
        sm_keys = [k for k in sys.modules if k == "statsmodels" or k.startswith("statsmodels.")]
        saved_sm = {k: sys.modules.pop(k) for k in sm_keys}

        module_key = "worldcup_playoff.models.ordered_logit"
        saved_module = sys.modules.pop(module_key, None)

        try:
            import worldcup_playoff.models.ordered_logit  # noqa: F401, re-import after eviction

            loaded_sm = [
                k for k in sys.modules if k == "statsmodels" or k.startswith("statsmodels.")
            ]
            assert loaded_sm == [], (
                "statsmodels must not be imported at module level; "
                f"found in sys.modules: {loaded_sm}"
            )
        finally:
            sys.modules.update(saved_sm)
            if saved_module is not None:
                sys.modules[module_key] = saved_module


# ---------------------------------------------------------------------------
# AC5 — Probability ordering: statsmodels [away, draw, home] → value object
# ---------------------------------------------------------------------------


class TestProbabilityOrdering:
    """Acceptance criterion: col-0 = away, col-1 = draw, col-2 = home in statsmodels output."""

    def _fitted_model(self):
        from worldcup_playoff.models.ordered_logit import OrderedLogitModel

        model = OrderedLogitModel()
        model.fit(_MinimalDataset(_make_train_df(n=120, seed=0)))  # type: ignore[arg-type]
        return model

    def test_when_elo_diff_strongly_positive_then_prob_home_exceeds_prob_away(self):
        """
        elo_diff >> 0 (home team dominant) → prob_home > prob_away.
        Validates [away, draw, home] column order is mapped correctly.
        """
        model = self._fitted_model()
        result = model.predict(pd.DataFrame({"elo_diff": [450.0]}))
        op = result[0]
        assert op.prob_home > op.prob_away, (
            "Large positive elo_diff must yield prob_home > prob_away; "
            f"got home={op.prob_home:.4f}, away={op.prob_away:.4f}"
        )

    def test_when_elo_diff_strongly_negative_then_prob_away_exceeds_prob_home(self):
        """
        elo_diff << 0 (away team dominant) → prob_away > prob_home.
        """
        model = self._fitted_model()
        result = model.predict(pd.DataFrame({"elo_diff": [-450.0]}))
        op = result[0]
        assert op.prob_away > op.prob_home, (
            "Large negative elo_diff must yield prob_away > prob_home; "
            f"got home={op.prob_home:.4f}, away={op.prob_away:.4f}"
        )

    def test_when_probabilities_returned_then_each_row_sums_to_one(self):
        model = self._fitted_model()
        probe = pd.DataFrame({"elo_diff": [250.0, 0.0, -250.0, 80.0, -80.0]})
        for op in model.predict(probe):
            total = op.prob_home + op.prob_draw + op.prob_away
            assert total == pytest.approx(1.0, abs=1e-6), (
                f"Probabilities must sum to 1.0, got {total} "
                f"(home={op.prob_home}, draw={op.prob_draw}, away={op.prob_away})"
            )

    def test_when_symmetric_elo_diff_then_home_and_away_probabilities_are_equal(self):
        """
        At elo_diff = 0 (perfectly matched teams), prob_home ≈ prob_away by symmetry.
        This catches a transposition that makes one side systematically wrong.
        """
        model = self._fitted_model()
        result = model.predict(pd.DataFrame({"elo_diff": [0.0]}))
        op = result[0]
        assert op.prob_home == pytest.approx(op.prob_away, abs=0.05), (
            "At elo_diff=0 the home and away win probabilities must be approximately equal; "
            f"got home={op.prob_home:.4f}, away={op.prob_away:.4f}"
        )


# ---------------------------------------------------------------------------
# Property-based tests (Hypothesis)
# Invariant derived from AC5: for *any* valid elo_diff batch, every OutcomeProbabilities
# has all three fields in [0, 1] and the row sums to 1.0.
# ---------------------------------------------------------------------------


@given(
    elo_diffs=st.lists(
        st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=15,
    )
)
@settings(max_examples=20, deadline=None)
def test_when_predict_called_with_any_valid_elo_diffs_then_probabilities_are_in_unit_interval(
    elo_diffs,
):
    """
    Invariant (AC5 / probability-row criterion): for any list of finite elo_diff values
    the returned OutcomeProbabilities always have all three fields in [0, 1] and sum to 1.0.

    Design note: the model is re-fitted inside the test so this property stays source-blind
    and self-contained; hypothesis max_examples is kept small to limit wall-clock time.
    """
    from worldcup_playoff.models.ordered_logit import OrderedLogitModel

    model = OrderedLogitModel()
    model.fit(_MinimalDataset(_make_train_df(n=80, seed=0)))  # type: ignore[arg-type]

    results = model.predict(pd.DataFrame({"elo_diff": elo_diffs}))

    assert len(results) == len(elo_diffs)
    for op in results:
        assert 0.0 <= op.prob_home <= 1.0
        assert 0.0 <= op.prob_draw <= 1.0
        assert 0.0 <= op.prob_away <= 1.0
        assert op.prob_home + op.prob_draw + op.prob_away == pytest.approx(1.0, abs=1e-6)
