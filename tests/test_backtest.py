"""
Source-blind example tests for the backtest_hybrid orchestrator.

The function under test (from worldcup_playoff.models.evaluation) slices played
WC matches by tournament year on time-aware (no-shuffle) splits, predicts W/D/L
with a hybrid model and a legacy classifier, and returns a per-year comparison
table that optionally includes a bookmaker baseline column.

All fixtures are synthetic — no network, no real data, no wall-clock dependencies.

Assumed interface (derived from spec, not implementation source):
    backtest_hybrid(
        matches  : pd.DataFrame,   # history rows with 'date', 'tournament', 'outcome',
                                   # plus feature columns the models consume
        hybrid   : Any,            # sklearn-style: fit(X, y) + predict_proba(X) → (N,3)
        legacy   : Any,            # same protocol
        odds     : pd.DataFrame | None = None,  # de-vigged bookmaker W/D/L probs
        years    : list[int] | None = None,     # tournament years to evaluate
    ) -> pd.DataFrame

Expected result columns:
    rps_hybrid    – mean RPS for the hybrid model (always present)
    rps_legacy    – mean RPS for the legacy classifier (always present)
    rps_bookmaker – mean RPS for the bookmaker baseline
                    (present ONLY when odds is not None)

Tournament year accessible either as a 'year' column or as the DataFrame index.

Outcome encoding: 0 = home win (W), 1 = draw (D), 2 = away win / loss (L).
"""

import numpy as np
import pandas as pd
import pytest

from worldcup_playoff.models.evaluation import backtest_hybrid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _ConstantModel:
    """
    Stub estimator: always returns a fixed probability vector, ignores input.
    fit() is a no-op so the orchestrator can call it without error.
    """

    def __init__(self, probs=(0.5, 0.3, 0.2)):
        self._probs = np.asarray(probs, dtype=float)

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        n = len(X)
        return np.tile(self._probs, (n, 1))


def _make_matches() -> pd.DataFrame:
    """
    Synthetic WC match rows for two tournament years (3 matches each).
    Includes 'elo_diff' and 'home_attack' as representative feature columns.
    Outcome: 0=W, 1=D, 2=L from home team's perspective.
    """
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2014-06-12",
                    "2014-06-13",
                    "2014-06-14",
                    "2022-11-20",
                    "2022-11-21",
                    "2022-11-22",
                ]
            ),
            "home_team": ["Brazil", "Spain", "Germany", "Qatar", "Senegal", "USA"],
            "away_team": [
                "Croatia",
                "Netherlands",
                "Portugal",
                "Ecuador",
                "Netherlands",
                "Wales",
            ],
            "tournament": ["FIFA World Cup"] * 6,
            "outcome": [0, 2, 0, 2, 1, 0],
            "elo_diff": [150.0, -50.0, 100.0, -200.0, 30.0, -10.0],
            "home_attack": [1.5, 1.2, 1.8, 0.5, 0.9, 1.0],
        }
    )


def _make_odds() -> pd.DataFrame:
    """
    Stub bookmaker odds aligned to the synthetic matches.
    Probabilities are already de-vigged (p_win + p_draw + p_loss == 1.0 per row).
    """
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2014-06-12",
                    "2014-06-13",
                    "2014-06-14",
                    "2022-11-20",
                    "2022-11-21",
                    "2022-11-22",
                ]
            ),
            "home_team": ["Brazil", "Spain", "Germany", "Qatar", "Senegal", "USA"],
            "away_team": [
                "Croatia",
                "Netherlands",
                "Portugal",
                "Ecuador",
                "Netherlands",
                "Wales",
            ],
            "p_win": [0.60, 0.30, 0.70, 0.20, 0.40, 0.35],
            "p_draw": [0.20, 0.30, 0.15, 0.25, 0.30, 0.30],
            "p_loss": [0.20, 0.40, 0.15, 0.55, 0.30, 0.35],
        }
    )


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def matches():
    return _make_matches()


@pytest.fixture()
def hybrid():
    return _ConstantModel(probs=(0.5, 0.3, 0.2))


@pytest.fixture()
def legacy():
    return _ConstantModel(probs=(0.4, 0.3, 0.3))


@pytest.fixture()
def odds():
    return _make_odds()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_when_two_tournament_years_provided_then_result_has_one_row_per_year(
    matches, hybrid, legacy
):
    """backtest_hybrid returns exactly one result row for each requested year."""
    result = backtest_hybrid(matches, hybrid, legacy, years=[2014, 2022])
    assert len(result) == 2


def test_when_two_tournament_years_provided_then_result_identifies_those_years(
    matches, hybrid, legacy
):
    """
    The requested years (2014, 2022) must appear in the result — either as a
    'year' column or as the integer index of the returned DataFrame.
    """
    result = backtest_hybrid(matches, hybrid, legacy, years=[2014, 2022])
    if "year" in result.columns:
        years_in_result = set(result["year"].astype(int).tolist())
    else:
        years_in_result = set(int(v) for v in result.index.tolist())
    assert 2014 in years_in_result
    assert 2022 in years_in_result


def test_when_backtest_runs_then_result_contains_hybrid_and_legacy_rps_columns(
    matches, hybrid, legacy
):
    """The comparison table always exposes rps_hybrid and rps_legacy."""
    result = backtest_hybrid(matches, hybrid, legacy, years=[2014, 2022])
    assert "rps_hybrid" in result.columns
    assert "rps_legacy" in result.columns


def test_when_odds_are_none_then_bookmaker_column_is_absent_but_hybrid_and_legacy_remain(
    matches, hybrid, legacy
):
    """
    Graceful degradation: passing odds=None must not raise, must still return
    rps_hybrid and rps_legacy, and must NOT include rps_bookmaker.
    """
    result = backtest_hybrid(matches, hybrid, legacy, odds=None, years=[2014, 2022])
    assert "rps_hybrid" in result.columns
    assert "rps_legacy" in result.columns
    assert "rps_bookmaker" not in result.columns


def test_when_odds_are_provided_then_bookmaker_rps_column_is_present(matches, hybrid, legacy, odds):
    """When odds are supplied the comparison table includes rps_bookmaker."""
    result = backtest_hybrid(matches, hybrid, legacy, odds=odds, years=[2014, 2022])
    assert "rps_bookmaker" in result.columns


def test_when_backtest_runs_with_odds_then_all_rps_values_are_in_unit_interval(
    matches, hybrid, legacy, odds
):
    """
    RPS ∈ [0, 1] is an analytic property of the metric.  All three reported
    RPS columns must satisfy this bound for every row in the result.
    """
    result = backtest_hybrid(matches, hybrid, legacy, odds=odds, years=[2014, 2022])
    for col in ("rps_hybrid", "rps_legacy", "rps_bookmaker"):
        assert col in result.columns, f"expected column '{col}' missing from result"
        values = result[col].dropna()
        assert (values >= 0).all(), f"{col}: found negative RPS value"
        assert (values <= 1).all(), f"{col}: found RPS value > 1"
