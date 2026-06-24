"""Source-blind example tests for issue #12: time-aware match dataset utility.

Tests are derived solely from the acceptance criteria — no implementation source
was read. Each test pins down one observable behaviour.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest
from hypothesis import given, strategies as st
from pandas import NA

from worldcup_playoff.models.dataset import (
    MatchDataset,
    add_targets,
    build_dataset,
    chronological_split,
    outcome_label,
    played_only,
)


# ---------------------------------------------------------------------------
# In-memory fixture helpers
# ---------------------------------------------------------------------------


def _played_df(n: int = 10) -> pd.DataFrame:
    """Chronologically ordered frame of n played matches."""
    rows = [
        {
            "date": f"2020-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}",
            "home_team": "Team A",
            "away_team": "Team B",
            "home_goals": i % 3,
            "away_goals": (i + 1) % 3,
            "home_elo": 1500.0,
            "away_elo": 1450.0,
            "elo_diff": 50.0,
        }
        for i in range(n)
    ]
    df = pd.DataFrame(rows)
    df["home_goals"] = df["home_goals"].astype("Int64")
    df["away_goals"] = df["away_goals"].astype("Int64")
    return df


def _mixed_df(n_played: int = 8, n_unplayed: int = 2) -> pd.DataFrame:
    """Played rows followed by unplayed (NA-goals) rows."""
    played = _played_df(n_played)
    unplayed_rows = [
        {
            "date": f"2030-06-{j + 1:02d}",
            "home_team": "Team C",
            "away_team": "Team D",
            "home_goals": NA,
            "away_goals": NA,
            "home_elo": 1480.0,
            "away_elo": 1490.0,
            "elo_diff": -10.0,
        }
        for j in range(n_unplayed)
    ]
    unplayed = pd.DataFrame(unplayed_rows)
    unplayed["home_goals"] = unplayed["home_goals"].astype("Int64")
    unplayed["away_goals"] = unplayed["away_goals"].astype("Int64")
    return pd.concat([played, unplayed], ignore_index=True)


_FEATURE_COLS = ["home_elo", "away_elo", "elo_diff"]


# ---------------------------------------------------------------------------
# outcome_label — ordered encoding
# ---------------------------------------------------------------------------


def test_when_home_wins_then_outcome_label_returns_2():
    assert outcome_label(2, 0) == 2


def test_when_away_wins_then_outcome_label_returns_0():
    assert outcome_label(0, 3) == 0


def test_when_draw_then_outcome_label_returns_1():
    assert outcome_label(1, 1) == 1


def test_when_both_goals_are_zero_then_outcome_label_returns_draw():
    assert outcome_label(0, 0) == 1


def test_when_home_wins_by_one_then_outcome_label_returns_2():
    assert outcome_label(1, 0) == 2


def test_when_away_wins_by_one_then_outcome_label_returns_0():
    assert outcome_label(0, 1) == 0


@given(
    home=st.integers(min_value=0, max_value=20),
    away=st.integers(min_value=0, max_value=20),
)
def test_when_outcome_label_called_then_result_is_in_valid_set(home: int, away: int) -> None:
    assert outcome_label(home, away) in {0, 1, 2}


@given(
    home=st.integers(min_value=0, max_value=20),
    away=st.integers(min_value=0, max_value=20),
)
def test_when_outcome_label_called_then_ordering_invariant_holds(home: int, away: int) -> None:
    result = outcome_label(home, away)
    if home > away:
        assert result == 2
    elif home == away:
        assert result == 1
    else:
        assert result == 0


# ---------------------------------------------------------------------------
# add_targets
# ---------------------------------------------------------------------------


def test_when_add_targets_called_then_y_outcome_column_is_added():
    df = _played_df()
    result = add_targets(df)
    assert "y_outcome" in result.columns


def test_when_add_targets_called_then_y_margin_column_is_added():
    df = _played_df()
    result = add_targets(df)
    assert "y_margin" in result.columns


def test_when_add_targets_called_then_y_outcome_dtype_is_nullable_int64():
    df = _played_df()
    result = add_targets(df)
    assert result["y_outcome"].dtype == pd.Int64Dtype()


def test_when_add_targets_called_then_y_outcome_values_are_in_valid_set():
    df = _played_df()
    result = add_targets(df)
    unique_vals = set(result["y_outcome"].dropna().unique())
    assert unique_vals.issubset({0, 1, 2})


def test_when_add_targets_called_then_y_margin_equals_home_minus_away_goals():
    df = _played_df(6)
    result = add_targets(df)
    expected = (result["home_goals"] - result["away_goals"]).astype("Int64")
    pd.testing.assert_series_equal(
        result["y_margin"].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


def test_when_add_targets_called_then_original_df_is_not_mutated():
    df = _played_df()
    cols_before = set(df.columns)
    add_targets(df)
    assert set(df.columns) == cols_before


def test_when_add_targets_called_then_row_count_is_unchanged():
    df = _played_df(8)
    assert len(add_targets(df)) == len(df)


@given(
    home=st.integers(min_value=0, max_value=15),
    away=st.integers(min_value=0, max_value=15),
)
def test_when_add_targets_called_then_y_margin_always_equals_home_minus_away(
    home: int, away: int
) -> None:
    df = pd.DataFrame(
        [
            {
                "date": "2020-01-01",
                "home_team": "A",
                "away_team": "B",
                "home_goals": home,
                "away_goals": away,
            }
        ]
    )
    df["home_goals"] = df["home_goals"].astype("Int64")
    df["away_goals"] = df["away_goals"].astype("Int64")
    result = add_targets(df)
    assert result["y_margin"].iloc[0] == home - away


# ---------------------------------------------------------------------------
# played_only
# ---------------------------------------------------------------------------


def test_when_played_only_called_then_rows_with_na_home_goals_are_dropped():
    df = _mixed_df(n_played=4, n_unplayed=2)
    result = played_only(df)
    assert result["home_goals"].isna().sum() == 0


def test_when_played_only_called_then_rows_with_na_away_goals_are_dropped():
    df = _mixed_df(n_played=4, n_unplayed=2)
    result = played_only(df)
    assert result["away_goals"].isna().sum() == 0


def test_when_played_only_called_on_fully_played_df_then_all_rows_are_retained():
    df = _played_df(6)
    assert len(played_only(df)) == len(df)


def test_when_played_only_called_then_only_played_rows_remain():
    df = _mixed_df(n_played=5, n_unplayed=3)
    result = played_only(df)
    assert len(result) == 5


def test_when_played_only_called_on_empty_df_then_empty_df_is_returned():
    df = pd.DataFrame(columns=["date", "home_goals", "away_goals"])
    df["home_goals"] = df["home_goals"].astype("Int64")
    df["away_goals"] = df["away_goals"].astype("Int64")
    result = played_only(df)
    assert len(result) == 0


# ---------------------------------------------------------------------------
# chronological_split
# ---------------------------------------------------------------------------


def _chrono_df(n: int = 12) -> pd.DataFrame:
    """Frame with strictly increasing ISO dates and played goals."""
    rows = [
        {
            "date": f"2020-01-{i + 1:02d}",
            "home_team": "A",
            "away_team": "B",
            "home_goals": 1,
            "away_goals": 0,
            "home_elo": 1500.0,
            "away_elo": 1450.0,
            "elo_diff": 50.0,
        }
        for i in range(n)
    ]
    df = pd.DataFrame(rows)
    df["home_goals"] = df["home_goals"].astype("Int64")
    df["away_goals"] = df["away_goals"].astype("Int64")
    return df


def test_when_chronological_split_then_train_max_date_lte_test_min_date():
    df = _chrono_df(10)
    train, test = chronological_split(df, test_size=0.2)
    assert train["date"].max() <= test["date"].min()


def test_when_chronological_split_then_row_counts_cover_all_input_rows():
    df = _chrono_df(10)
    train, test = chronological_split(df, test_size=0.2)
    assert len(train) + len(test) == len(df)


def test_when_chronological_split_then_test_size_is_floor_fraction_of_total():
    n = 10
    test_size = 0.3
    df = _chrono_df(n)
    _, test = chronological_split(df, test_size=test_size)
    assert len(test) == math.floor(n * test_size)


def test_when_chronological_split_called_twice_then_same_train_is_returned():
    df = _chrono_df(10)
    train1, _ = chronological_split(df, test_size=0.2)
    train2, _ = chronological_split(df, test_size=0.2)
    pd.testing.assert_frame_equal(train1.reset_index(drop=True), train2.reset_index(drop=True))


def test_when_chronological_split_called_twice_then_same_test_is_returned():
    df = _chrono_df(10)
    _, test1 = chronological_split(df, test_size=0.2)
    _, test2 = chronological_split(df, test_size=0.2)
    pd.testing.assert_frame_equal(test1.reset_index(drop=True), test2.reset_index(drop=True))


def test_when_chronological_split_then_last_train_date_not_later_than_first_test_date():
    df = _chrono_df(12)
    train, test = chronological_split(df, test_size=0.25)
    assert train["date"].iloc[-1] <= test["date"].iloc[0]


# ---------------------------------------------------------------------------
# build_dataset
# ---------------------------------------------------------------------------


def test_when_build_dataset_called_then_returns_match_dataset_instance():
    df = _mixed_df(n_played=8, n_unplayed=2)
    ds = build_dataset(df, test_size=0.25, feature_cols=_FEATURE_COLS)
    assert isinstance(ds, MatchDataset)


def test_when_build_dataset_called_then_train_contains_no_na_home_goals():
    df = _mixed_df(n_played=8, n_unplayed=2)
    ds = build_dataset(df, test_size=0.25, feature_cols=_FEATURE_COLS)
    assert not ds.train["home_goals"].isna().any()


def test_when_build_dataset_called_then_train_contains_no_na_away_goals():
    df = _mixed_df(n_played=8, n_unplayed=2)
    ds = build_dataset(df, test_size=0.25, feature_cols=_FEATURE_COLS)
    assert not ds.train["away_goals"].isna().any()


def test_when_build_dataset_called_then_test_contains_no_na_home_goals():
    df = _mixed_df(n_played=8, n_unplayed=2)
    ds = build_dataset(df, test_size=0.25, feature_cols=_FEATURE_COLS)
    assert not ds.test["home_goals"].isna().any()


def test_when_build_dataset_called_then_test_contains_no_na_away_goals():
    df = _mixed_df(n_played=8, n_unplayed=2)
    ds = build_dataset(df, test_size=0.25, feature_cols=_FEATURE_COLS)
    assert not ds.test["away_goals"].isna().any()


def test_when_build_dataset_called_then_train_and_test_row_counts_sum_to_played_count():
    n_played = 10
    df = _mixed_df(n_played=n_played, n_unplayed=3)
    ds = build_dataset(df, test_size=0.2, feature_cols=_FEATURE_COLS)
    assert len(ds.train) + len(ds.test) == n_played


def test_when_build_dataset_called_then_feature_cols_are_stored_on_dataset():
    df = _mixed_df()
    ds = build_dataset(df, test_size=0.25, feature_cols=_FEATURE_COLS)
    assert ds.feature_cols == _FEATURE_COLS


def test_when_build_dataset_called_then_train_max_date_lte_test_min_date():
    df = _mixed_df(n_played=10, n_unplayed=2)
    ds = build_dataset(df, test_size=0.2, feature_cols=_FEATURE_COLS)
    assert ds.train["date"].max() <= ds.test["date"].min()


def test_when_build_dataset_called_then_match_dataset_is_frozen():
    df = _mixed_df()
    ds = build_dataset(df, test_size=0.25, feature_cols=_FEATURE_COLS)
    with pytest.raises((AttributeError, TypeError)):
        ds.train = pd.DataFrame()  # type: ignore[misc]
