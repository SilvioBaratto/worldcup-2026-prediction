"""
Source-blind example tests for issue #10 criteria 2 and 3:

Criterion 2:
    live_fixtures_to_df(state) maps a list of LiveMatch to the exact martj42
    results column set/dtypes (passes validate_results_df), sets
    TOURNAMENT="FIFA World Cup", <NA> goals for remaining fixtures, and
    drops None-team placeholder slots.

Criterion 3:
    The entire suite runs with no network — the live path is exercised only
    via constructed TournamentState/LiveMatch objects (mirroring the pattern
    established in tests/test_live_adapter.py).

Tests are derived solely from the acceptance criteria and requirements.md.
No implementation source was read; these tests describe the contract the
implementation must satisfy (Red phase of TDD).

Choices recorded where the spec is silent:
  - live_fixtures_to_df resides in worldcup_playoff.data.live (the live
    adapter module, alongside LiveMatch and TournamentState).
  - The function accepts a TournamentState and processes
    state.remaining_group_fixtures — the list of not-yet-played group
    matches returned by LiveTournamentAdapter.tournament_state().
  - Martj42 results schema uses uppercase column names:
    DATE, HOME_TEAM, AWAY_TEAM, HOME_GOALS (nullable Int64),
    AWAY_GOALS (nullable Int64), TOURNAMENT, NEUTRAL.
    (Consistent with requirements.md Data Contracts and test_feature_builder.py.)
  - validate_results_df is the schema-validator exported from
    worldcup_playoff.data.martj42_loader (the module that owns the
    martj42 internal schema definition).
  - All LiveMatch objects constructed here have neutral=True (WC2026 is
    played in neutral venues per requirements.md).
  - "None-team placeholder slots" = LiveMatch with home_team=None (LAST_32
    knockout slots not yet filled; observed in test_live_adapter.py fixture
    _LAST_32_NULL_TEAMS_MATCH).

Criteria skipped (NOT VERIFIABLE per oracle):
  - No-key path produces a usable WC2026 frame without any API key.
  - Deterministic given the seed.
  - All tests pass / SOLID quality gates.
"""

from __future__ import annotations

import pandas as pd
from hypothesis import given, settings, strategies as st

# Expected martj42 results column set (uppercase per requirements.md Data Contracts)
_RESULTS_COLS = frozenset(
    {"DATE", "HOME_TEAM", "AWAY_TEAM", "HOME_GOALS", "AWAY_GOALS", "TOURNAMENT", "NEUTRAL"}
)

# ---------------------------------------------------------------------------
# Fixture constructors — no network I/O; derived from spec only
# ---------------------------------------------------------------------------


def _live_match(
    home_team: str | None = "Brazil",
    away_team: str | None = "Argentina",
    home_goals: int | None = None,
    away_goals: int | None = None,
    date: str = "2026-06-28",
    status: str = "SCHEDULED",
    neutral: bool = True,
    match_id: int = 1,
):
    """Construct a single LiveMatch with sensible defaults for a scheduled WC2026 fixture."""
    from worldcup_playoff.data.live import LiveMatch

    return LiveMatch(
        id=match_id,
        home_team=home_team,
        away_team=away_team,
        home_goals=home_goals,
        away_goals=away_goals,
        date=date,
        status=status,
        neutral=neutral,
    )


def _null_slot(match_id: int = 99, date: str = "2026-07-01"):
    """LAST_32-style knockout slot where teams are not yet known (both names None)."""
    return _live_match(
        home_team=None,
        away_team=None,
        match_id=match_id,
        date=date,
    )


def _state(*matches):
    """Build a TournamentState whose remaining_group_fixtures contains the given matches."""
    from worldcup_playoff.data.live import TournamentState

    return TournamentState(
        played=[],
        remaining_group_fixtures=list(matches),
        standings=[],
    )


# ---------------------------------------------------------------------------
# Criterion 2a — output contains the exact martj42 results column set
# ---------------------------------------------------------------------------


class TestLiveFixturesToDfSchema:
    """live_fixtures_to_df must return a DataFrame with every martj42 results column."""

    def test_when_single_fixture_converted_then_date_column_is_present(self) -> None:
        from worldcup_playoff.data.live import live_fixtures_to_df

        result = live_fixtures_to_df(_state(_live_match()))
        assert "DATE" in result.columns

    def test_when_single_fixture_converted_then_home_team_column_is_present(self) -> None:
        from worldcup_playoff.data.live import live_fixtures_to_df

        result = live_fixtures_to_df(_state(_live_match()))
        assert "HOME_TEAM" in result.columns

    def test_when_single_fixture_converted_then_away_team_column_is_present(self) -> None:
        from worldcup_playoff.data.live import live_fixtures_to_df

        result = live_fixtures_to_df(_state(_live_match()))
        assert "AWAY_TEAM" in result.columns

    def test_when_single_fixture_converted_then_home_goals_column_is_present(self) -> None:
        from worldcup_playoff.data.live import live_fixtures_to_df

        result = live_fixtures_to_df(_state(_live_match()))
        assert "HOME_GOALS" in result.columns

    def test_when_single_fixture_converted_then_away_goals_column_is_present(self) -> None:
        from worldcup_playoff.data.live import live_fixtures_to_df

        result = live_fixtures_to_df(_state(_live_match()))
        assert "AWAY_GOALS" in result.columns

    def test_when_single_fixture_converted_then_tournament_column_is_present(self) -> None:
        from worldcup_playoff.data.live import live_fixtures_to_df

        result = live_fixtures_to_df(_state(_live_match()))
        assert "TOURNAMENT" in result.columns

    def test_when_single_fixture_converted_then_neutral_column_is_present(self) -> None:
        from worldcup_playoff.data.live import live_fixtures_to_df

        result = live_fixtures_to_df(_state(_live_match()))
        assert "NEUTRAL" in result.columns

    def test_when_multiple_fixtures_converted_then_all_required_columns_are_present(self) -> None:
        from worldcup_playoff.data.live import live_fixtures_to_df

        state = _state(
            _live_match("Spain", "Portugal", match_id=1),
            _live_match("France", "Germany", match_id=2),
        )
        result = live_fixtures_to_df(state)
        missing = _RESULTS_COLS - set(result.columns)
        assert not missing, f"Missing columns: {missing}"

    def test_when_empty_state_converted_then_result_is_empty_dataframe(self) -> None:
        from worldcup_playoff.data.live import live_fixtures_to_df

        result = live_fixtures_to_df(_state())
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Criterion 2b — TOURNAMENT is always "FIFA World Cup"
# ---------------------------------------------------------------------------


class TestLiveFixturesToDfTournamentColumn:
    """Every row in the output must have TOURNAMENT == 'FIFA World Cup'."""

    def test_when_remaining_fixture_converted_then_tournament_is_fifa_world_cup(self) -> None:
        from worldcup_playoff.data.live import live_fixtures_to_df

        result = live_fixtures_to_df(_state(_live_match("USA", "Canada")))
        assert result.iloc[0]["TOURNAMENT"] == "FIFA World Cup"

    def test_when_multiple_fixtures_converted_then_all_have_tournament_fifa_world_cup(self) -> None:
        from worldcup_playoff.data.live import live_fixtures_to_df

        state = _state(
            _live_match("Spain", "Portugal", match_id=1),
            _live_match("France", "Germany", match_id=2),
            _live_match("Brazil", "Mexico", match_id=3),
        )
        result = live_fixtures_to_df(state)
        assert (result["TOURNAMENT"] == "FIFA World Cup").all()


# ---------------------------------------------------------------------------
# Criterion 2c — HOME_GOALS / AWAY_GOALS are <NA> for remaining fixtures
# ---------------------------------------------------------------------------


class TestLiveFixturesToDfNaGoals:
    """Remaining (unplayed) fixtures must carry <NA> for HOME_GOALS and AWAY_GOALS."""

    def test_when_scheduled_fixture_converted_then_home_goals_are_na(self) -> None:
        from worldcup_playoff.data.live import live_fixtures_to_df

        result = live_fixtures_to_df(_state(_live_match("Brazil", "France", status="SCHEDULED")))
        assert pd.isna(result.iloc[0]["HOME_GOALS"])

    def test_when_scheduled_fixture_converted_then_away_goals_are_na(self) -> None:
        from worldcup_playoff.data.live import live_fixtures_to_df

        result = live_fixtures_to_df(_state(_live_match("Brazil", "France", status="SCHEDULED")))
        assert pd.isna(result.iloc[0]["AWAY_GOALS"])

    def test_when_timed_fixture_converted_then_home_goals_are_na(self) -> None:
        from worldcup_playoff.data.live import live_fixtures_to_df

        result = live_fixtures_to_df(_state(_live_match("Argentina", "Spain", status="TIMED")))
        assert pd.isna(result.iloc[0]["HOME_GOALS"])

    def test_when_timed_fixture_converted_then_away_goals_are_na(self) -> None:
        from worldcup_playoff.data.live import live_fixtures_to_df

        result = live_fixtures_to_df(_state(_live_match("Argentina", "Spain", status="TIMED")))
        assert pd.isna(result.iloc[0]["AWAY_GOALS"])

    def test_when_multiple_scheduled_fixtures_then_all_goals_are_na(self) -> None:
        from worldcup_playoff.data.live import live_fixtures_to_df

        state = _state(
            _live_match("Japan", "Senegal", match_id=1),
            _live_match("Morocco", "Croatia", match_id=2),
        )
        result = live_fixtures_to_df(state)
        assert result["HOME_GOALS"].isna().all()
        assert result["AWAY_GOALS"].isna().all()


# ---------------------------------------------------------------------------
# Criterion 2d — None-team placeholder slots are dropped
# ---------------------------------------------------------------------------


class TestLiveFixturesToDfNullTeamDrop:
    """LiveMatch objects whose home_team is None must be absent from the output."""

    def test_when_state_has_one_null_slot_then_it_is_dropped(self) -> None:
        from worldcup_playoff.data.live import live_fixtures_to_df

        state = _state(
            _live_match("Brazil", "Mexico", match_id=1),
            _null_slot(match_id=2),
        )
        result = live_fixtures_to_df(state)
        assert len(result) == 1
        assert result.iloc[0]["HOME_TEAM"] == "Brazil"

    def test_when_all_slots_are_null_then_output_is_empty(self) -> None:
        from worldcup_playoff.data.live import live_fixtures_to_df

        state = _state(_null_slot(1), _null_slot(2), _null_slot(3))
        result = live_fixtures_to_df(state)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_when_mixed_valid_and_null_slots_then_row_count_equals_valid_count(self) -> None:
        from worldcup_playoff.data.live import live_fixtures_to_df

        state = _state(
            _live_match("USA", "Canada", match_id=1),
            _null_slot(match_id=2),
            _live_match("England", "Croatia", match_id=3),
            _null_slot(match_id=4),
        )
        result = live_fixtures_to_df(state)
        assert len(result) == 2

    def test_when_mixed_slots_then_only_valid_team_names_appear(self) -> None:
        from worldcup_playoff.data.live import live_fixtures_to_df

        state = _state(
            _live_match("Spain", "Portugal", match_id=1),
            _null_slot(match_id=2),
        )
        result = live_fixtures_to_df(state)
        assert "Spain" in result["HOME_TEAM"].values
        assert result["HOME_TEAM"].notna().all()

    def test_when_output_produced_then_no_row_has_null_home_team(self) -> None:
        from worldcup_playoff.data.live import live_fixtures_to_df

        state = _state(
            _live_match("France", "Germany", match_id=1),
            _null_slot(match_id=2),
            _live_match("Japan", "Senegal", match_id=3),
        )
        result = live_fixtures_to_df(state)
        assert result["HOME_TEAM"].notna().all()

    def test_when_output_produced_then_no_row_has_null_away_team(self) -> None:
        from worldcup_playoff.data.live import live_fixtures_to_df

        state = _state(
            _live_match("France", "Germany", match_id=1),
            _null_slot(match_id=2),
            _live_match("Japan", "Senegal", match_id=3),
        )
        result = live_fixtures_to_df(state)
        assert result["AWAY_TEAM"].notna().all()


# ---------------------------------------------------------------------------
# Criterion 2e — output passes validate_results_df (schema contract)
# ---------------------------------------------------------------------------


class TestLiveFixturesToDfValidation:
    def test_when_valid_fixtures_converted_then_validate_results_df_does_not_raise(self) -> None:
        """
        live_fixtures_to_df output must satisfy the martj42 results schema validator.
        Derived from criterion: 'passes validate_results_df'.
        """
        from worldcup_playoff.data.live import live_fixtures_to_df
        from worldcup_playoff.data.martj42_loader import validate_results_df

        state = _state(
            _live_match("Brazil", "Argentina", match_id=1),
            _live_match("France", "Germany", match_id=2),
            _live_match("Spain", "Portugal", match_id=3),
        )
        result = live_fixtures_to_df(state)
        validate_results_df(result)  # must not raise

    def test_when_state_with_null_slots_converted_then_validate_results_df_does_not_raise(
        self,
    ) -> None:
        """Null-slot filtering must not break schema compliance."""
        from worldcup_playoff.data.live import live_fixtures_to_df
        from worldcup_playoff.data.martj42_loader import validate_results_df

        state = _state(
            _live_match("USA", "Canada", match_id=1),
            _null_slot(match_id=2),
        )
        result = live_fixtures_to_df(state)
        validate_results_df(result)


# ---------------------------------------------------------------------------
# Property-based tests — invariants from criterion 2
# ---------------------------------------------------------------------------


@given(
    n_valid=st.integers(min_value=0, max_value=6),
    n_null=st.integers(min_value=0, max_value=6),
)
@settings(max_examples=30)
def test_when_any_mix_of_valid_and_null_slots_then_row_count_equals_valid_count(
    n_valid: int,
    n_null: int,
) -> None:
    """
    Invariant: for ANY combination of valid and null-team LiveMatch objects,
    live_fixtures_to_df drops exactly the null-team slots, leaving precisely
    n_valid rows in the output.

    Derived from criterion: 'drops None-team placeholder slots.'
    """
    from worldcup_playoff.data.live import LiveMatch, TournamentState, live_fixtures_to_df

    valid_matches = [
        LiveMatch(
            id=i,
            home_team=f"HomeTeam{i}",
            away_team=f"AwayTeam{i}",
            home_goals=None,
            away_goals=None,
            date="2026-07-01",
            status="SCHEDULED",
            neutral=True,
        )
        for i in range(n_valid)
    ]
    null_matches = [
        LiveMatch(
            id=n_valid + j,
            home_team=None,
            away_team=None,
            home_goals=None,
            away_goals=None,
            date="2026-07-01",
            status="SCHEDULED",
            neutral=True,
        )
        for j in range(n_null)
    ]
    state = TournamentState(
        played=[],
        remaining_group_fixtures=valid_matches + null_matches,
        standings=[],
    )
    result = live_fixtures_to_df(state)

    assert len(result) == n_valid


@given(n_matches=st.integers(min_value=1, max_value=8))
@settings(max_examples=20)
def test_when_any_count_of_valid_fixtures_converted_then_tournament_is_always_wc(
    n_matches: int,
) -> None:
    """
    Invariant: TOURNAMENT == 'FIFA World Cup' for every row in the output,
    regardless of how many valid fixtures are converted.

    Derived from criterion: 'sets TOURNAMENT="FIFA World Cup".'
    """
    from worldcup_playoff.data.live import LiveMatch, TournamentState, live_fixtures_to_df

    matches = [
        LiveMatch(
            id=i,
            home_team=f"Team{i}A",
            away_team=f"Team{i}B",
            home_goals=None,
            away_goals=None,
            date="2026-07-01",
            status="SCHEDULED",
            neutral=True,
        )
        for i in range(n_matches)
    ]
    state = TournamentState(
        played=[],
        remaining_group_fixtures=matches,
        standings=[],
    )
    result = live_fixtures_to_df(state)

    assert (result["TOURNAMENT"] == "FIFA World Cup").all()
