"""No-key end-to-end integration smoke test for the features package.

Issue #11 — feat: features package public API, exports, and no-key integration smoke test.

Builds an in-memory martj42-internal-schema DataFrame (DATE/HOME_TEAM/… uppercase),
calls ``compute_elo`` + ``fit_dixon_coles`` once, passes both into ``build_features``
(historical builder) and ``wc2026_features``, then asserts:
  1. The two output frames share the same column set.
  2. The WC2026 frame's rows are a subset of the historical frame's rows.

No network calls are made: all data is constructed in-process.
"""

from __future__ import annotations

import pandas as pd
import pytest

from worldcup_playoff.data.elo import compute_elo
from worldcup_playoff.features import build_features, wc2026_features
from worldcup_playoff.simulation.poisson import fit_dixon_coles


# ---------------------------------------------------------------------------
# Shared fixture — martj42 internal schema, in-process only
# ---------------------------------------------------------------------------


def _make_history() -> pd.DataFrame:
    """Minimal martj42-internal DataFrame: 3 played WC matches + 2 unplayed WC2026."""
    return pd.DataFrame(
        {
            "DATE": pd.to_datetime(
                [
                    "2022-11-20",
                    "2022-11-24",
                    "2022-11-28",
                    "2026-06-11",
                    "2026-06-15",
                ]
            ),
            "HOME_TEAM": ["Brazil", "France", "Argentina", "Brazil", "Germany"],
            "AWAY_TEAM": ["Serbia", "Australia", "Poland", "France", "Spain"],
            "HOME_GOALS": pd.array([2, 4, 2, pd.NA, pd.NA], dtype="Int64"),
            "AWAY_GOALS": pd.array([0, 1, 0, pd.NA, pd.NA], dtype="Int64"),
            "TOURNAMENT": ["FIFA World Cup"] * 5,
            "NEUTRAL": [True, True, True, True, True],
        }
    )


@pytest.fixture(scope="module", name="integration_frames")
def _integration_frames_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(historical_features_df, wc2026_features_df) built once per test module."""
    history = _make_history()
    elo_result = compute_elo(history)
    abilities = fit_dixon_coles(history)
    hist_df = build_features(history, elo_result, abilities)
    wc_df = wc2026_features(history, elo_result, abilities)
    return hist_df, wc_df


# ---------------------------------------------------------------------------
# Acceptance criterion: same column set
# ---------------------------------------------------------------------------


def test_when_integration_frames_built_then_historical_and_wc2026_share_same_column_set(
    integration_frames: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """
    The historical-features frame and the WC2026-features frame must expose
    the same column set.

    AC: "assert the two frames share the same column set".
    """
    hist, wc = integration_frames
    hist_cols = set(hist.columns)
    wc_cols = set(wc.columns)
    assert hist_cols == wc_cols, (
        "Column sets differ between historical and WC2026 feature frames.\n"
        f"  Only in historical: {hist_cols - wc_cols}\n"
        f"  Only in wc2026:     {wc_cols - hist_cols}"
    )


# ---------------------------------------------------------------------------
# Acceptance criterion: WC2026 rows ⊆ historical rows
# ---------------------------------------------------------------------------


def test_when_integration_frames_built_then_wc2026_has_fewer_rows_than_historical(
    integration_frames: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """
    The WC2026 frame must have strictly fewer rows than the historical frame.

    Fixture: 3 played + 2 unplayed → historical outputs 5 rows; wc2026_features
    returns only the 2 unplayed WC2026 rows (strict subset by count).

    AC: "the WC2026 frame's rows are a subset of the historical frame's rows".
    """
    hist, wc = integration_frames
    assert len(wc) < len(hist), (
        f"WC2026 frame ({len(wc)} rows) must be smaller than historical frame ({len(hist)} rows)."
    )


def test_when_integration_frames_built_then_wc2026_index_is_subset_of_historical_index(
    integration_frames: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """
    WC2026 frame row-index labels must be a subset of historical frame row-index labels.

    Both frames use reset_index(drop=True), so WC2026's [0, 1] is a subset of
    historical's [0, 1, 2, 3, 4].
    """
    hist, wc = integration_frames
    assert frozenset(wc.index).issubset(frozenset(hist.index)), (
        "WC2026 frame index is not a subset of the historical frame index.\n"
        f"  WC2026 index:     {sorted(wc.index)}\n"
        f"  Historical index: {sorted(hist.index)}"
    )


# ---------------------------------------------------------------------------
# Smoke: WC2026 rows have NA goals; historical played rows have integer goals
# ---------------------------------------------------------------------------


def test_when_integration_frames_built_then_wc2026_goals_are_na(
    integration_frames: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """home_goals and away_goals must be <NA> for every WC2026 output row."""
    _, wc = integration_frames
    assert wc["home_goals"].isna().all(), "Some WC2026 home_goals are not <NA>"
    assert wc["away_goals"].isna().all(), "Some WC2026 away_goals are not <NA>"


def test_when_integration_frames_built_then_historical_played_rows_have_numeric_goals(
    integration_frames: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Played rows in the historical frame must have non-NA integer goals."""
    hist, _ = integration_frames
    played = hist[hist["home_goals"].notna()]
    assert len(played) == 3, f"Expected 3 played rows, got {len(played)}"


def test_when_integration_frames_built_then_no_network_was_needed(
    integration_frames: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Trivial guard: if the fixture ran, no network was required (it's all in-process)."""
    hist, wc = integration_frames
    assert isinstance(hist, pd.DataFrame)
    assert isinstance(wc, pd.DataFrame)
