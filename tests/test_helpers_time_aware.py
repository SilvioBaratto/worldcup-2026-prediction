"""Tests for Issue #8: time-aware recent-form, rest-days, and goal-difference helpers.

BDD naming: 'when X, Y is returned'.  All DataFrames are built in-memory — no
network / I/O.  Tests exercise the no-forward-leakage contract explicitly.

Acceptance criteria covered:
  1  sort_chronological: stable ascending sort, na_position='last', reset index
  1a sort_chronological byte-identical to Elo engine's sort (alignment guard)
  2  recent_form: 0.0 neutral default; 3.0 PPG for single 3-0 win
  3  no-leakage: later match does not change recent_form(before_idx=i)
  4  rest_days: exact calendar delta; None on first appearance
  5  away-only: team appearing only as away still accumulates form/goal-diff
  6  NA goals skipped by form/goal-diff; rest_days convention does not raise
  +  goal_difference: windowed net GD from team's perspective
  +  Hypothesis property: sort_chronological idempotent
  +  Hypothesis property: no-leakage holds for any valid frame and cutoff
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from worldcup_playoff.data.elo import SORT_COLS
from worldcup_playoff.features.timeaware import (
    goal_difference,
    recent_form,
    rest_days,
    sort_chronological,
)

# ---------------------------------------------------------------------------
# Shared frame factory
# ---------------------------------------------------------------------------

_W, _D, _L = 5, 1, 365  # window, half_life defaults match FeatureBuildConfig


def _frame(*rows: tuple) -> pd.DataFrame:
    """Build a minimal matches DataFrame from (home, away, date_or_None, hg_or_None, ag_or_None)."""
    return pd.DataFrame(
        [
            {
                "HOME_TEAM": home,
                "AWAY_TEAM": away,
                "DATE": pd.to_datetime(d) if d is not None else pd.NaT,
                "HOME_GOALS": hg,
                "AWAY_GOALS": ag,
                "TOURNAMENT": "Friendly",
                "NEUTRAL": False,
            }
            for home, away, d, hg, ag in rows
        ]
    )


# ---------------------------------------------------------------------------
# Criterion 1 — sort_chronological
# ---------------------------------------------------------------------------


class TestSortChronological:
    """sort_chronological sorts ascending, puts NaT last, and resets the index."""

    def test_when_dates_are_mixed_then_result_is_sorted_ascending(self) -> None:
        df = _frame(
            ("Brazil", "Germany", "2018-07-15", 1, 1),
            ("France", "Croatia", "2018-07-10", 4, 2),
            ("England", "Belgium", "2018-07-14", 0, 2),
        )
        result = sort_chronological(df)
        dates = result["DATE"].dropna().tolist()
        assert dates == sorted(dates)

    def test_when_nat_date_present_then_nat_row_is_last(self) -> None:
        df = _frame(
            ("Brazil", "Germany", None, 1, 1),
            ("France", "Croatia", "2018-07-10", 4, 2),
            ("England", "Belgium", "2018-07-14", 0, 2),
        )
        result = sort_chronological(df)
        assert pd.isna(result["DATE"].iloc[-1])

    def test_when_sorted_then_index_is_reset_to_zero_based_range(self) -> None:
        df = _frame(
            ("France", "Croatia", "2018-07-10", 4, 2),
            ("Brazil", "Germany", "2018-07-05", 1, 1),
        )
        result = sort_chronological(df)
        assert list(result.index) == list(range(len(result)))

    def test_when_dates_equal_then_sort_is_stable_on_team_name(self) -> None:
        """Same date: sort falls through to HOME_TEAM alphabetically ('A' < 'C')."""
        df = _frame(
            ("A", "B", "2020-01-01", 1, 0),
            ("C", "D", "2020-01-01", 2, 1),
        )
        result = sort_chronological(df)
        homes = list(result["HOME_TEAM"])
        assert homes.index("A") < homes.index("C")

    def test_when_compared_to_elo_sort_then_frames_are_byte_identical(self) -> None:
        """sort_chronological must reproduce the Elo engine's sort exactly.

        Any divergence here would misalign the sorted frame with
        EloResult.match_diffs and silently corrupt downstream features.
        """
        df = _frame(
            ("Brazil", "Germany", "2018-07-15", 1, 1),
            ("France", "Croatia", "2018-07-10", 4, 2),
            ("England", "Belgium", "2018-07-14", 0, 2),
            ("Spain", "Portugal", None, 0, 0),
        )
        expected = (
            df.assign(_d=pd.to_datetime(df["DATE"], errors="coerce"))
            .sort_values(SORT_COLS, kind="stable", na_position="last")
            .drop(columns=["_d"])
            .reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(sort_chronological(df), expected)


# ---------------------------------------------------------------------------
# Criterion 2 — recent_form
# ---------------------------------------------------------------------------


class TestRecentForm:
    """recent_form returns 0.0 neutral default or decay-weighted PPG."""

    def test_when_no_prior_matches_then_recent_form_is_zero(self) -> None:
        df = _frame(("Brazil", "Germany", "2022-12-18", 3, 0))
        assert recent_form("Brazil", 0, df, window=5, half_life_days=365.0) == 0.0

    def test_when_single_win_then_recent_form_is_positive(self) -> None:
        df = _frame(
            ("Brazil", "Germany", "2022-12-10", 3, 0),  # idx 0 — win
            ("Brazil", "France", "2022-12-18", 0, 1),  # idx 1 — current
        )
        assert recent_form("Brazil", 1, df, window=5, half_life_days=365.0) > 0.0

    def test_when_single_three_nil_win_then_ppg_equals_three(self) -> None:
        """With one prior match, weighted_sum / sum_of_weights = 3·w / w = 3.0."""
        df = _frame(
            ("Argentina", "England", "2022-12-10", 3, 0),  # idx 0 — win
            ("Argentina", "France", "2022-12-18", 1, 0),  # idx 1 — current
        )
        assert recent_form("Argentina", 1, df, window=5, half_life_days=365.0) == pytest.approx(3.0)

    def test_when_prior_match_is_draw_then_ppg_equals_one(self) -> None:
        df = _frame(
            ("England", "France", "2022-12-10", 1, 1),  # idx 0 — draw
            ("England", "Spain", "2022-12-18", 0, 0),  # idx 1 — current
        )
        assert recent_form("England", 1, df, window=5, half_life_days=365.0) == pytest.approx(1.0)

    def test_when_prior_match_is_loss_then_ppg_equals_zero(self) -> None:
        df = _frame(
            ("England", "Brazil", "2022-12-10", 0, 3),  # idx 0 — loss
            ("England", "Spain", "2022-12-18", 0, 0),  # idx 1 — current
        )
        assert recent_form("England", 1, df, window=5, half_life_days=365.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Criterion 3 — no-leakage
# ---------------------------------------------------------------------------


class TestNoLeakage:
    """recent_form(before_idx=i) must not change when a later row is appended."""

    def test_when_later_match_appended_then_form_at_earlier_cutoff_is_unchanged(self) -> None:
        base = [
            ("Germany", "Brazil", "2022-12-01", 1, 0),  # idx 0
            ("Germany", "France", "2022-12-10", 0, 1),  # idx 1
            ("Germany", "Spain", "2022-12-18", 2, 0),  # idx 2  ← cut-off
        ]
        df_original = _frame(*base)
        form_before = recent_form("Germany", 2, df_original, window=5, half_life_days=365.0)

        extended = base + [("Germany", "Italy", "2022-12-25", 3, 2)]
        df_extended = _frame(*extended)
        form_after = recent_form("Germany", 2, df_extended, window=5, half_life_days=365.0)

        assert form_before == pytest.approx(form_after)


# ---------------------------------------------------------------------------
# Criterion 4 — rest_days
# ---------------------------------------------------------------------------


class TestRestDays:
    """rest_days returns None on first appearance; exact calendar delta otherwise."""

    def test_when_first_appearance_then_rest_days_is_none(self) -> None:
        df = _frame(("France", "Croatia", "2018-07-15", 4, 2))
        assert rest_days("France", 0, df) is None

    def test_when_two_matches_five_days_apart_then_rest_days_is_five(self) -> None:
        df = _frame(
            ("France", "Belgium", "2018-07-10", 1, 0),  # idx 0
            ("France", "Croatia", "2018-07-15", 4, 2),  # idx 1
        )
        assert rest_days("France", 1, df) == 5

    def test_when_exact_nine_day_gap_then_rest_days_is_nine(self) -> None:
        date_a, date_b = date(2022, 11, 20), date(2022, 11, 29)
        df = _frame(
            ("Argentina", "Saudi Arabia", date_a.isoformat(), 1, 2),
            ("Argentina", "Mexico", date_b.isoformat(), 2, 0),
        )
        assert rest_days("Argentina", 1, df) == (date_b - date_a).days


# ---------------------------------------------------------------------------
# Criterion 5 — away-only symmetry
# ---------------------------------------------------------------------------


class TestAwayOnlySymmetry:
    """A team appearing only as away still accumulates form and goal-difference."""

    def test_when_team_is_only_away_then_recent_form_is_positive_after_win(self) -> None:
        df = _frame(
            ("Brazil", "Germany", "2022-12-10", 1, 3),  # idx 0 — Germany wins away
            ("France", "Germany", "2022-12-18", 0, 0),  # idx 1 — current match
        )
        result = recent_form("Germany", 1, df, window=5, half_life_days=365.0)
        assert result == pytest.approx(3.0)

    def test_when_team_is_only_away_then_goal_difference_reflects_away_goals(self) -> None:
        df = _frame(
            ("Brazil", "Germany", "2022-12-10", 1, 3),  # idx 0 — Germany away: +2 GD
            ("France", "Germany", "2022-12-18", 0, 0),  # idx 1 — current
        )
        result = goal_difference("Germany", 1, df, window=5)
        assert result == pytest.approx(2.0)  # 3 scored − 1 conceded


# ---------------------------------------------------------------------------
# Criterion 6 — NA goals skipped
# ---------------------------------------------------------------------------


class TestNAGoalsSkipped:
    """Unplayed fixtures (<NA> goals) are excluded from form / goal-difference."""

    def test_when_only_prior_is_unplayed_then_recent_form_is_zero(self) -> None:
        df = _frame(
            ("Spain", "Germany", "2022-12-01", None, None),  # idx 0 — unplayed
            ("Spain", "Brazil", "2022-12-18", 1, 0),  # idx 1 — current
        )
        assert recent_form("Spain", 1, df, window=5, half_life_days=365.0) == 0.0

    def test_when_only_prior_is_unplayed_then_goal_difference_is_zero(self) -> None:
        df = _frame(
            ("Spain", "Germany", "2022-12-01", None, None),
            ("Spain", "Brazil", "2022-12-18", 1, 0),
        )
        assert goal_difference("Spain", 1, df, window=5) == 0.0

    def test_when_na_goals_prior_exists_then_rest_days_does_not_raise(self) -> None:
        """rest_days uses dates regardless of goal data; must not raise."""
        df = _frame(
            ("Spain", "Germany", "2022-12-01", None, None),
            ("Spain", "Brazil", "2022-12-18", 1, 0),
        )
        result = rest_days("Spain", 1, df)
        assert result is None or isinstance(result, int)


# ---------------------------------------------------------------------------
# goal_difference additional tests
# ---------------------------------------------------------------------------


class TestGoalDifference:
    """goal_difference computes windowed net GD from team's perspective."""

    def test_when_no_prior_matches_then_goal_difference_is_zero(self) -> None:
        df = _frame(("France", "Brazil", "2022-12-18", 2, 1))
        assert goal_difference("France", 0, df, window=5) == 0.0

    def test_when_home_win_three_nil_then_goal_difference_is_three(self) -> None:
        df = _frame(
            ("Brazil", "Germany", "2022-12-10", 3, 0),  # idx 0 — Brazil +3
            ("Brazil", "France", "2022-12-18", 1, 0),  # idx 1 — current
        )
        assert goal_difference("Brazil", 1, df, window=5) == pytest.approx(3.0)

    def test_when_window_limits_matches_then_only_recent_rows_counted(self) -> None:
        df = _frame(
            ("Brazil", "Germany", "2022-11-01", 5, 0),  # idx 0 — +5 (outside window=1)
            ("Brazil", "France", "2022-12-01", 1, 2),  # idx 1 — -1 (inside window=1)
            ("Brazil", "Spain", "2022-12-18", 0, 0),  # idx 2 — current
        )
        assert goal_difference("Brazil", 2, df, window=1) == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# Hypothesis property-based tests
# ---------------------------------------------------------------------------

_DATE_STRAT = st.dates(min_value=date(2000, 1, 1), max_value=date(2030, 12, 31))


@given(
    rows=st.lists(
        st.tuples(
            st.sampled_from(["Germany", "Brazil", "France"]),
            st.sampled_from(["Spain", "England", "Argentina"]),
            _DATE_STRAT,
        ),
        min_size=1,
        max_size=20,
    )
)
@settings(max_examples=100)
def test_when_sort_applied_twice_then_result_is_idempotent(rows: list) -> None:
    """sort_chronological(sort_chronological(df)) == sort_chronological(df)."""
    df = pd.DataFrame(
        [
            {
                "HOME_TEAM": h,
                "AWAY_TEAM": a,
                "DATE": pd.Timestamp(d.isoformat()),
                "HOME_GOALS": 1,
                "AWAY_GOALS": 0,
                "TOURNAMENT": "Friendly",
                "NEUTRAL": False,
            }
            for h, a, d in rows
        ]
    )
    once = sort_chronological(df)
    twice = sort_chronological(once)
    pd.testing.assert_frame_equal(once.reset_index(drop=True), twice.reset_index(drop=True))


@given(
    rows=st.lists(
        st.tuples(
            st.sampled_from(["Germany", "Brazil", "TeamX"]),
            st.sampled_from(["France", "Spain", "TeamY"]),
            _DATE_STRAT,
            st.integers(min_value=0, max_value=5),
            st.integers(min_value=0, max_value=5),
        ),
        min_size=3,
        max_size=15,
    )
)
@settings(max_examples=80)
def test_when_future_row_appended_then_form_at_earlier_cutoff_is_unchanged(rows: list) -> None:
    """No-leakage invariant holds for any frame and any valid before_idx."""
    df = pd.DataFrame(
        [
            {
                "HOME_TEAM": h,
                "AWAY_TEAM": a,
                "DATE": pd.Timestamp(d.isoformat()),
                "HOME_GOALS": hg,
                "AWAY_GOALS": ag,
                "TOURNAMENT": "Friendly",
                "NEUTRAL": False,
            }
            for h, a, d, hg, ag in rows
        ]
    )
    cutoff = max(1, len(df) // 2)
    form_original = recent_form("Germany", cutoff, df, window=5, half_life_days=365.0)

    extra = pd.DataFrame(
        [
            {
                "HOME_TEAM": "Germany",
                "AWAY_TEAM": "Spain",
                "DATE": pd.Timestamp("2099-06-01"),
                "HOME_GOALS": 5,
                "AWAY_GOALS": 0,
                "TOURNAMENT": "Friendly",
                "NEUTRAL": False,
            }
        ]
    )
    df_ext = pd.concat([df, extra], ignore_index=True)
    form_extended = recent_form("Germany", cutoff, df_ext, window=5, half_life_days=365.0)

    assert form_original == pytest.approx(form_extended)
