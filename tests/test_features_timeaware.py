"""
Source-blind example tests for worldcup_playoff/features/timeaware.py — Issue #35.

All tests are derived from the acceptance-criteria text only (Red-phase TDD).
No implementation source was read during authoring.

Invariant choices documented per test where the criterion is ambiguous:
- "played match" = row where both HOME_GOALS and AWAY_GOALS are non-NaN.
- recent_form returns points on the 3/1/0 scale; a single prior match returns
  exactly 3.0/1.0/0.0, which implies weighted-average (not weighted-sum) form
  so that the single weight cancels in numerator/denominator.
- goal_difference from a team's perspective: HOME_GOALS − AWAY_GOALS when home,
  AWAY_GOALS − HOME_GOALS when away.
- before_idx is 0-based positional (iloc-style), matching the reset index that
  sort_chronological produces.
"""

import numpy as np
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
# DataFrame builders
# ---------------------------------------------------------------------------


def _match(date, home, away, home_goals=None, away_goals=None):
    return {
        "DATE": pd.Timestamp(date) if date is not None else pd.NaT,
        "HOME_TEAM": home,
        "AWAY_TEAM": away,
        "HOME_GOALS": float(home_goals) if home_goals is not None else np.nan,
        "AWAY_GOALS": float(away_goals) if away_goals is not None else np.nan,
    }


def _df(*rows):
    return pd.DataFrame(list(rows))


# ---------------------------------------------------------------------------
# sort_chronological
# ---------------------------------------------------------------------------


def test_when_nat_date_is_present_then_nat_rows_appear_last():
    df = _df(
        _match(None, "A", "B"),
        _match("2020-01-01", "C", "D", 2, 1),
        _match("2019-06-01", "E", "F", 0, 0),
    )
    result = sort_chronological(df)
    assert pd.isna(result.iloc[-1]["DATE"])


def test_when_sort_chronological_applied_then_index_is_zero_based_consecutive():
    df = _df(
        _match("2021-01-03", "X", "Y", 1, 0),
        _match("2020-05-10", "A", "B", 0, 2),
    )
    result = sort_chronological(df)
    assert list(result.index) == list(range(len(result)))


def test_when_sort_chronological_applied_twice_then_result_is_unchanged():
    df = _df(
        _match("2021-07-01", "G", "H", 3, 1),
        _match("2019-03-15", "I", "J", 0, 0),
        _match(None, "K", "L", 1, 2),
    )
    once = sort_chronological(df)
    twice = sort_chronological(once)
    pd.testing.assert_frame_equal(once, twice)


def test_when_sort_chronological_applied_then_ordering_is_byte_identical_to_sort_cols_ascending_nat_last():
    """Output must match: assign _d via to_datetime(errors='coerce'), sort by SORT_COLS, drop _d, reset index.

    SORT_COLS[0] is '_d' — a synthetic column derived from DATE with errors='coerce' so
    invalid/NaT dates sort last.  The expected frame must create _d the same way before
    sorting; otherwise sort_values raises KeyError on '_d'.
    """
    df = _df(
        _match("2021-07-01", "G", "H", 3, 1),
        _match("2019-03-15", "I", "J", 0, 0),
        _match(None, "K", "L", 1, 2),
        _match("2020-01-01", "M", "N", 2, 2),
    )
    result = sort_chronological(df)
    expected = (
        df.assign(_d=pd.to_datetime(df["DATE"], errors="coerce"))
        .sort_values(SORT_COLS, ascending=True, na_position="last")
        .drop(columns=["_d"])
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(result, expected)


def test_when_sort_chronological_applied_then_synthetic_d_column_is_absent_from_result():
    """The '_d' work column must be dropped before returning — it must not leak into the frame."""
    df = _df(
        _match("2020-01-01", "A", "B", 1, 0),
        _match(None, "C", "D", 2, 1),
    )
    result = sort_chronological(df)
    assert "_d" not in result.columns


# ---------------------------------------------------------------------------
# recent_form
# ---------------------------------------------------------------------------


def test_when_no_prior_played_match_then_recent_form_returns_zero():
    df = sort_chronological(_df(_match("2020-01-01", "Spain", "France", 2, 1)))
    assert recent_form("Spain", 0, df) == pytest.approx(0.0)


def test_when_single_prior_home_win_then_recent_form_returns_three():
    df = sort_chronological(
        _df(
            _match("2020-01-01", "Spain", "France", 2, 1),  # win at idx 0
            _match("2020-01-02", "Spain", "Brazil", 1, 0),  # reference at idx 1
        )
    )
    assert recent_form("Spain", 1, df) == pytest.approx(3.0)


def test_when_single_prior_home_draw_then_recent_form_returns_one():
    df = sort_chronological(
        _df(
            _match("2020-01-01", "Spain", "France", 1, 1),
            _match("2020-01-02", "Spain", "Brazil", 2, 0),
        )
    )
    assert recent_form("Spain", 1, df) == pytest.approx(1.0)


def test_when_single_prior_home_loss_then_recent_form_returns_zero():
    df = sort_chronological(
        _df(
            _match("2020-01-01", "Spain", "France", 0, 2),
            _match("2020-01-02", "Spain", "Brazil", 1, 0),
        )
    )
    assert recent_form("Spain", 1, df) == pytest.approx(0.0)


def test_when_single_prior_away_win_then_recent_form_returns_three():
    df = sort_chronological(
        _df(
            _match("2020-01-01", "France", "Spain", 0, 1),  # Spain wins away
            _match("2020-01-02", "Spain", "Brazil", 1, 0),
        )
    )
    assert recent_form("Spain", 1, df) == pytest.approx(3.0)


def test_when_rows_after_before_idx_exist_then_recent_form_excludes_them():
    """Rows at index >= before_idx must be invisible to the computation."""
    df = sort_chronological(
        _df(
            _match("2020-01-01", "Spain", "France", 2, 1),  # prior win   idx 0
            _match("2020-01-02", "Spain", "Brazil", 1, 0),  # reference   idx 1
            _match("2020-01-03", "Spain", "Germany", 0, 3),  # future loss idx 2 — must be ignored
        )
    )
    assert recent_form("Spain", 1, df) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# rest_days
# ---------------------------------------------------------------------------


def test_when_no_prior_match_then_rest_days_returns_none():
    df = sort_chronological(_df(_match("2020-01-10", "Germany", "France", 1, 0)))
    assert rest_days("Germany", 0, df) is None


def test_when_one_prior_home_match_then_rest_days_returns_exact_calendar_delta():
    df = sort_chronological(
        _df(
            _match("2020-01-01", "Germany", "France", 2, 0),
            _match("2020-01-11", "Germany", "Brazil", 0, 1),  # 10 days later
        )
    )
    assert rest_days("Germany", 1, df) == 10


def test_when_one_prior_away_match_then_rest_days_returns_exact_calendar_delta():
    df = sort_chronological(
        _df(
            _match("2020-01-01", "France", "Germany", 0, 2),  # Germany plays away
            _match("2020-01-16", "Germany", "Brazil", 0, 1),  # 15 days later
        )
    )
    assert rest_days("Germany", 1, df) == 15


def test_when_multiple_prior_matches_then_rest_days_uses_most_recent_prior_match():
    """Criterion says 'most recent prior dated match' — not the oldest."""
    df = sort_chronological(
        _df(
            _match("2020-01-01", "Germany", "France", 2, 0),  # older
            _match("2020-01-05", "Germany", "Italy", 1, 1),  # 4 days before reference
            _match("2020-01-09", "Germany", "Brazil", 3, 0),  # reference
        )
    )
    assert rest_days("Germany", 2, df) == 4


# ---------------------------------------------------------------------------
# goal_difference
# ---------------------------------------------------------------------------


def test_when_no_prior_played_match_then_goal_difference_returns_zero():
    df = sort_chronological(_df(_match("2020-01-01", "Brazil", "France", 2, 1)))
    assert goal_difference("Brazil", 0, df) == pytest.approx(0.0)


def test_when_single_home_win_then_goal_difference_is_home_minus_away():
    df = sort_chronological(
        _df(
            _match("2020-01-01", "Brazil", "France", 3, 1),  # +2 for Brazil
            _match("2020-01-02", "Brazil", "Germany", 0, 0),  # reference
        )
    )
    assert goal_difference("Brazil", 1, df) == pytest.approx(2.0)


def test_when_single_away_win_then_goal_difference_is_away_minus_home():
    df = sort_chronological(
        _df(
            _match("2020-01-01", "France", "Brazil", 1, 3),  # Brazil: 3−1 = +2 as away
            _match("2020-01-02", "Brazil", "Germany", 0, 0),
        )
    )
    assert goal_difference("Brazil", 1, df) == pytest.approx(2.0)


def test_when_single_home_loss_then_goal_difference_is_negative():
    df = sort_chronological(
        _df(
            _match("2020-01-01", "Brazil", "France", 0, 2),  # −2 for Brazil
            _match("2020-01-02", "Brazil", "Germany", 0, 0),
        )
    )
    assert goal_difference("Brazil", 1, df) == pytest.approx(-2.0)


def test_when_matches_exceed_window_then_only_last_window_played_matches_are_summed():
    """window=2 → last 2 prior played matches; oldest (idx 0, +1) must be excluded."""
    df = sort_chronological(
        _df(
            _match("2020-01-01", "Brazil", "France", 1, 0),  # +1  (excluded by window=2)
            _match("2020-01-02", "Brazil", "Argentina", 2, 0),  # +2
            _match("2020-01-03", "Brazil", "Spain", 3, 0),  # +3
            _match("2020-01-04", "Brazil", "Germany", 0, 0),  # reference
        )
    )
    assert goal_difference("Brazil", 3, df, window=2) == pytest.approx(5.0)


def test_when_unplayed_match_is_present_then_it_is_excluded_from_goal_difference():
    """NaN-goals rows are unplayed fixtures and must not count as played matches."""
    df = sort_chronological(
        _df(
            _match("2020-01-01", "Brazil", "France", 2, 0),  # played: +2
            _match("2020-01-02", "Brazil", "Argentina", None, None),  # unplayed: skip
            _match("2020-01-03", "Brazil", "Germany", 0, 0),  # reference
        )
    )
    assert goal_difference("Brazil", 2, df) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# No-leakage — appending a later row must not change values at a fixed before_idx
# ---------------------------------------------------------------------------


def test_when_later_row_appended_then_recent_form_at_fixed_before_idx_is_unchanged():
    base = sort_chronological(
        _df(
            _match("2020-01-01", "Spain", "France", 2, 1),
            _match("2020-06-01", "Spain", "Brazil", 1, 0),  # reference at idx 1
        )
    )
    form_before = recent_form("Spain", 1, base)

    extra = pd.DataFrame([_match("2025-01-01", "Spain", "Germany", 0, 3)])
    extended = sort_chronological(pd.concat([base, extra], ignore_index=True))

    assert recent_form("Spain", 1, extended) == pytest.approx(form_before)


def test_when_later_row_appended_then_rest_days_at_fixed_before_idx_is_unchanged():
    base = sort_chronological(
        _df(
            _match("2020-01-01", "Spain", "France", 2, 1),
            _match("2020-06-01", "Spain", "Brazil", 1, 0),
        )
    )
    rest_before = rest_days("Spain", 1, base)

    extra = pd.DataFrame([_match("2025-01-01", "Spain", "Germany", 0, 3)])
    extended = sort_chronological(pd.concat([base, extra], ignore_index=True))

    assert rest_days("Spain", 1, extended) == rest_before


def test_when_later_row_appended_then_goal_difference_at_fixed_before_idx_is_unchanged():
    base = sort_chronological(
        _df(
            _match("2020-01-01", "Spain", "France", 3, 0),
            _match("2020-06-01", "Spain", "Brazil", 1, 0),
        )
    )
    gd_before = goal_difference("Spain", 1, base)

    extra = pd.DataFrame([_match("2025-01-01", "Spain", "Germany", 0, 3)])
    extended = sort_chronological(pd.concat([base, extra], ignore_index=True))

    assert goal_difference("Spain", 1, extended) == pytest.approx(gd_before)


def test_when_team_appears_only_as_away_then_recent_form_accumulates():
    """Away-only teams must not be silently ignored — their wins count."""
    df = sort_chronological(
        _df(
            _match("2020-01-01", "France", "Tahiti", 1, 2),  # Tahiti away win
            _match("2020-03-01", "Brazil", "Tahiti", 0, 1),  # Tahiti away win
            _match("2020-06-01", "Germany", "Tahiti", 1, 0),  # reference at idx 2
        )
    )
    assert recent_form("Tahiti", 2, df) > 0.0


def test_when_team_appears_only_as_away_then_goal_difference_reflects_away_perspective():
    df = sort_chronological(
        _df(
            _match("2020-01-01", "France", "Tahiti", 1, 3),  # Tahiti: 3−1 = +2
            _match("2020-06-01", "Germany", "Tahiti", 0, 0),  # reference
        )
    )
    assert goal_difference("Tahiti", 1, df) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Property-based tests (Hypothesis)
# ---------------------------------------------------------------------------


def _build_df_from_dates(date_strings):
    rows = [_match(d, f"T{i}", f"T{i + 1}", i % 4, (i + 1) % 4) for i, d in enumerate(date_strings)]
    if not rows:
        return pd.DataFrame(columns=["DATE", "HOME_TEAM", "AWAY_TEAM", "HOME_GOALS", "AWAY_GOALS"])
    return pd.DataFrame(rows)


_DATE_STRATEGY = st.one_of(
    st.dates(
        min_value=pd.Timestamp("1950-01-01").date(),
        max_value=pd.Timestamp("2030-12-31").date(),
    ).map(str),
    st.just(None),
)


@given(st.lists(_DATE_STRATEGY, min_size=0, max_size=15))
def test_when_sort_chronological_applied_twice_then_result_is_idempotent(date_strings):
    """sort_chronological is idempotent for any mix of dated and NaT rows."""
    df = _build_df_from_dates(date_strings)
    once = sort_chronological(df)
    twice = sort_chronological(once)
    pd.testing.assert_frame_equal(once, twice)


@given(st.integers(min_value=1, max_value=8))
@settings(max_examples=40)
def test_when_future_row_appended_then_recent_form_at_fixed_before_idx_is_stable(n_prior):
    """No-leakage invariant: appending a later match never changes form at a fixed index."""
    prior_rows = [
        _match(f"2020-{(i % 12) + 1:02d}-01", "Spain", "France", 2, i % 3) for i in range(n_prior)
    ]
    df = sort_chronological(pd.DataFrame(prior_rows))
    idx = len(df) - 1

    form_original = recent_form("Spain", idx, df)

    extra = pd.DataFrame([_match("2099-12-31", "Spain", "Germany", 0, 5)])
    extended = sort_chronological(pd.concat([df, extra], ignore_index=True))

    assert recent_form("Spain", idx, extended) == pytest.approx(form_original)


@given(st.integers(min_value=1, max_value=8))
@settings(max_examples=40)
def test_when_future_row_appended_then_goal_difference_at_fixed_before_idx_is_stable(n_prior):
    """No-leakage invariant: appending a later match never changes GD at a fixed index."""
    prior_rows = [
        _match(f"2020-{(i % 12) + 1:02d}-01", "Spain", "France", i % 5, (i + 1) % 5)
        for i in range(n_prior)
    ]
    df = sort_chronological(pd.DataFrame(prior_rows))
    idx = len(df) - 1

    gd_original = goal_difference("Spain", idx, df)

    extra = pd.DataFrame([_match("2099-12-31", "Spain", "Germany", 0, 9)])
    extended = sort_chronological(pd.concat([df, extra], ignore_index=True))

    assert goal_difference("Spain", idx, extended) == pytest.approx(gd_original)
