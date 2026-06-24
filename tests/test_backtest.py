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


# ---------------------------------------------------------------------------
# run_backtest — orchestration tests
# Issue #48 acceptance criteria:
#   (1) run_backtest loads match odds across cfg.odds.seasons via
#       odds.load_match_odds + to_match_probs, concatenates, passes odds=
#       into backtest_hybrid.
#   (2) When odds load empty (scraper blocked), odds=None is passed so
#       rps_bookmaker is cleanly omitted (no crash).
#   (5) New test asserts rps_bookmaker appears with stubbed match odds.
#
# Patch paths assume worldcup_playoff.models.evaluation imports the odds
# module as:  from worldcup_playoff.data import odds  (module-qualified calls).
# If the implementation uses `from worldcup_playoff.data.odds import …`
# adjust the patch targets to worldcup_playoff.models.evaluation.<name>.
# ---------------------------------------------------------------------------

from unittest.mock import MagicMock, patch  # noqa: E402

from worldcup_playoff.models.evaluation import run_backtest  # noqa: E402

# ── Minimal features DataFrame returned by the _read_features mock ──────────
# run_backtest calls _read_features(".")  which reads dataset/features.csv.
# That file is not present in CI / unit-test runs, so every run_backtest test
# must patch it.  The exact shape doesn't matter because backtest_hybrid is
# also mocked in all of these tests.

_MINIMAL_FEATURES = pd.DataFrame(
    {
        "date": pd.to_datetime(["2014-06-12", "2022-11-20"]),
        "home_team": ["Brazil", "Qatar"],
        "away_team": ["Croatia", "Ecuador"],
        "tournament": ["FIFA World Cup"] * 2,
        "outcome": [0, 2],
        "elo_diff": [100.0, -200.0],
    }
)

_PATCH_FEATURES = "worldcup_playoff.models.evaluation._read_features"


def _make_raw_odds_df(year: int) -> pd.DataFrame:
    """Minimal DataFrame as load_match_odds returns (de-vigged p_win/p_draw/p_loss)."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime([f"{year}-06-12", f"{year}-06-13"]),
            "home_team": ["Brazil", "Germany"],
            "away_team": ["Argentina", "France"],
            "p_win": [0.40, 0.35],
            "p_draw": [0.30, 0.30],
            "p_loss": [0.30, 0.35],
        }
    )


def _cfg(seasons=None):
    cfg = MagicMock()
    cfg.odds.seasons = seasons if seasons is not None else [2014, 2018, 2022]
    return cfg


# -- Criterion 1: load_match_odds is called once per configured season --------


def test_when_seasons_configured_then_load_match_odds_is_called_once_per_season():
    """
    run_backtest must call load_match_odds for every entry in cfg.odds.seasons,
    passing the tournament key "wc{year}" and the config object.
    """
    seasons = [2014, 2018, 2022]
    cfg = _cfg(seasons=seasons)

    with (
        patch(_PATCH_FEATURES, return_value=_MINIMAL_FEATURES),
        patch("worldcup_playoff.data.odds.load_match_odds") as mock_load,
        patch("worldcup_playoff.models.evaluation.backtest_hybrid") as mock_bh,
    ):
        # load_match_odds(tournament: str, config: Any) — two positional args
        mock_load.side_effect = lambda t, c: _make_raw_odds_df(int(t[2:]))
        mock_bh.return_value = pd.DataFrame(
            {"rps_hybrid": [0.20, 0.22, 0.18], "rps_legacy": [0.25, 0.27, 0.24]},
            index=seasons,
        )

        run_backtest(cfg)

    assert mock_load.call_count == len(seasons)
    for s in seasons:
        # Years are mapped to tournament keys before calling load_match_odds
        mock_load.assert_any_call(f"wc{s}", cfg)


# -- Criterion 1: concatenated odds are passed as odds= to backtest_hybrid ----


def test_when_odds_loaded_then_backtest_hybrid_receives_non_none_odds_keyword():
    """
    After loading all seasons, run_backtest must pass a non-None DataFrame as the
    odds= keyword argument to backtest_hybrid.
    """
    cfg = _cfg(seasons=[2018, 2022])

    with (
        patch(_PATCH_FEATURES, return_value=_MINIMAL_FEATURES),
        patch("worldcup_playoff.data.odds.load_match_odds") as mock_load,
        patch("worldcup_playoff.models.evaluation.backtest_hybrid") as mock_bh,
    ):
        mock_load.side_effect = lambda t, c: _make_raw_odds_df(int(t[2:]))
        mock_bh.return_value = pd.DataFrame(
            {"rps_hybrid": [0.20, 0.18], "rps_legacy": [0.25, 0.24]},
            index=[2018, 2022],
        )

        run_backtest(cfg)

    assert mock_bh.called, "backtest_hybrid must be called by run_backtest"
    _, kwargs = mock_bh.call_args
    odds_arg = kwargs.get("odds")
    assert odds_arg is not None, (
        "backtest_hybrid must receive a non-None odds= argument when odds are available"
    )


# -- Criterion 2: empty odds → odds=None, no crash ----------------------------


def test_when_all_seasons_return_empty_odds_then_run_backtest_does_not_raise():
    """
    Scraper-blocked scenario (empty DataFrame for every season) must not crash.
    """
    cfg = _cfg(seasons=[2018])

    with (
        patch(_PATCH_FEATURES, return_value=_MINIMAL_FEATURES),
        patch("worldcup_playoff.data.odds.load_match_odds", return_value=pd.DataFrame()),
        patch(
            "worldcup_playoff.models.evaluation.backtest_hybrid",
            return_value=pd.DataFrame({"rps_hybrid": [0.20], "rps_legacy": [0.25]}, index=[2018]),
        ),
    ):
        run_backtest(cfg)  # must not raise


def test_when_all_seasons_return_empty_odds_then_backtest_hybrid_receives_none_for_odds():
    """
    When every season yields an empty DataFrame, run_backtest must pass odds=None
    so backtest_hybrid can cleanly omit rps_bookmaker.
    """
    cfg = _cfg(seasons=[2018])

    with (
        patch(_PATCH_FEATURES, return_value=_MINIMAL_FEATURES),
        patch("worldcup_playoff.data.odds.load_match_odds", return_value=pd.DataFrame()),
        patch("worldcup_playoff.models.evaluation.backtest_hybrid") as mock_bh,
    ):
        mock_bh.return_value = pd.DataFrame(
            {"rps_hybrid": [0.20], "rps_legacy": [0.25]}, index=[2018]
        )
        run_backtest(cfg)

    _, kwargs = mock_bh.call_args
    assert kwargs.get("odds") is None, (
        "When all odds DataFrames are empty, backtest_hybrid must be called with odds=None"
    )


# -- Criterion 5: rps_bookmaker appears in output when stubbed odds provided --


def test_when_run_backtest_called_with_stubbed_match_odds_then_rps_bookmaker_appears_in_output():
    """
    Criterion 5 — stubbing load_match_odds to return valid data must result in the
    final output containing rps_bookmaker.
    """
    seasons = [2018]
    cfg = _cfg(seasons=seasons)

    bh_result = pd.DataFrame(
        {
            "rps_hybrid": [0.19],
            "rps_legacy": [0.23],
            "rps_bookmaker": [0.17],
        },
        index=seasons,
    )

    with (
        patch(_PATCH_FEATURES, return_value=_MINIMAL_FEATURES),
        patch("worldcup_playoff.data.odds.load_match_odds") as mock_load,
        patch("worldcup_playoff.models.evaluation.backtest_hybrid", return_value=bh_result),
    ):
        mock_load.return_value = _make_raw_odds_df(2018)

        result = run_backtest(cfg)

    assert result is not None
    if isinstance(result, pd.DataFrame):
        assert "rps_bookmaker" in result.columns, (
            "rps_bookmaker must appear in run_backtest output when stubbed match odds are provided"
        )
    else:
        for year_data in result.values():
            assert "rps_bookmaker" in year_data, (
                "rps_bookmaker must appear in run_backtest output when stubbed match odds are provided"
            )
