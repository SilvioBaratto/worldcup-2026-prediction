"""Source-blind spec tests for issue #28: live WC2026 adapter.

Tests are derived exclusively from the acceptance criteria and requirements.md.
No implementation source was read during authoring — this is the Red phase of TDD.

Skipped criteria (per oracle):
  - "Adapter never crashes when the key/API is missing — falls back offline"
      → NOT VERIFIABLE: no concrete runtime assertion inferable without reading src.
  - "All tests pass" — boilerplate suite gate; no per-criterion assertion.
  - "SOLID, clean code" — subjective prose; no concrete runtime/unit assertion.

Assumed module layout (derived from requirements.md project-structure section):
  worldcup_playoff/data/live.py  — LiveMatch, TableRow, GroupStanding,
                                    TournamentState, LiveTournamentAdapter,
                                    fetch_tournament_state, build_state_from_results,
                                    live_fixtures_to_df
  worldcup_playoff/data/client.py — FootballClient (pre-existing)

Crosswalk direction (from test_crosswalk_spec.py alias table):
  football-data.org alias  →  canonical (martj42) name
  "Türkiye"               →  "Turkey"
  "IR Iran"               →  "Iran"
  "USA"                   →  "United States"
  "Korea Republic"        →  "South Korea"
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from worldcup_playoff.data.live import (
    GroupStanding,
    LiveMatch,
    LiveTournamentAdapter,
    TableRow,
    TournamentState,
    build_state_from_results,
    fetch_tournament_state,
    live_fixtures_to_df,
)

# ---------------------------------------------------------------------------
# Raw v4 API payloads — shapes taken verbatim from requirements.md data contracts
# ---------------------------------------------------------------------------

_RAW_MATCH_FINISHED: dict = {
    "id": 1001,
    "utcDate": "2026-06-15T18:00:00Z",
    "status": "FINISHED",
    "stage": "GROUP_STAGE",
    "group": "GROUP_A",
    "matchday": 1,
    "homeTeam": {"id": 759, "name": "Germany"},
    "awayTeam": {"id": 762, "name": "Japan"},
    "score": {
        "winner": "HOME_TEAM",
        "duration": "REGULAR",
        "fullTime": {"home": 2, "away": 1},
        "halfTime": {"home": 1, "away": 0},
    },
}

_RAW_MATCH_SCHEDULED: dict = {
    "id": 1002,
    "utcDate": "2026-06-20T21:00:00Z",
    "status": "SCHEDULED",
    "stage": "GROUP_STAGE",
    "group": "GROUP_A",
    "matchday": 2,
    "homeTeam": {"id": 759, "name": "Germany"},
    "awayTeam": {"id": 763, "name": "Spain"},
    "score": {
        "winner": None,
        "duration": "REGULAR",
        "fullTime": {"home": None, "away": None},
        "halfTime": {"home": None, "away": None},
    },
}

_RAW_TABLE_ROW: dict = {
    "position": 1,
    "team": {"id": 759, "name": "Germany"},
    "playedGames": 2,
    "form": "WW",
    "won": 2,
    "draw": 0,
    "lost": 0,
    "points": 6,
    "goalsFor": 5,
    "goalsAgainst": 1,
    "goalDifference": 4,
}

_RAW_GROUP_STANDING: dict = {
    "stage": "GROUP_STAGE",
    "type": "TOTAL",
    "group": "GROUP_A",
    "table": [_RAW_TABLE_ROW],
}

# martj42 column order, used in every DataFrame fixture
_MARTJ42_COLS = [
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "city",
    "country",
    "neutral",
]

# The seven martj42 schema columns that live_fixtures_to_df must produce
_MARTJ42_SCHEMA = frozenset(_MARTJ42_COLS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _df(*rows: tuple) -> pd.DataFrame:
    """Build a martj42-schema DataFrame from a list of tuples."""
    return pd.DataFrame(rows, columns=_MARTJ42_COLS)


def _wc2026_row(home: str, away: str, hs, aws, *, played: bool = True) -> tuple:
    """Return one WC2026 fixture row; played=False sets scores to None."""
    h_score = float(hs) if played else None
    a_score = float(aws) if played else None
    return (
        "2026-06-15",
        home,
        away,
        h_score,
        a_score,
        "FIFA World Cup",
        "Dallas",
        "United States",
        True,
    )


# ============================================================================
# AC-1 — LiveMatch: flatten v4 nested JSON, ignore extra fields, crosswalk
# ============================================================================


def test_when_livematch_validated_then_home_team_name_is_flat_field() -> None:
    m = LiveMatch.model_validate(_RAW_MATCH_FINISHED)
    assert m.home_team == "Germany"


def test_when_livematch_validated_then_away_team_name_is_flat_field() -> None:
    m = LiveMatch.model_validate(_RAW_MATCH_FINISHED)
    assert m.away_team == "Japan"


def test_when_livematch_validated_then_fulltime_home_score_is_flat_field() -> None:
    m = LiveMatch.model_validate(_RAW_MATCH_FINISHED)
    assert m.home_score == 2


def test_when_livematch_validated_then_fulltime_away_score_is_flat_field() -> None:
    m = LiveMatch.model_validate(_RAW_MATCH_FINISHED)
    assert m.away_score == 1


def test_when_livematch_validated_then_status_is_accessible() -> None:
    m = LiveMatch.model_validate(_RAW_MATCH_FINISHED)
    assert m.status == "FINISHED"


def test_when_livematch_validated_with_extra_field_then_it_is_silently_ignored() -> None:
    raw = {**_RAW_MATCH_FINISHED, "undocumentedKey": "should_be_dropped"}
    m = LiveMatch.model_validate(raw)
    assert not hasattr(m, "undocumentedKey")


def test_when_livematch_validated_for_unplayed_match_then_scores_are_none() -> None:
    m = LiveMatch.model_validate(_RAW_MATCH_SCHEDULED)
    assert m.home_score is None
    assert m.away_score is None


def test_when_livematch_home_team_is_ir_iran_then_crosswalk_normalizes_to_iran() -> None:
    """Crosswalk: "IR Iran" (football-data.org alias) → "Iran" (canonical)."""
    raw = {**_RAW_MATCH_FINISHED, "homeTeam": {"id": 1, "name": "IR Iran"}}
    m = LiveMatch.model_validate(raw)
    assert m.home_team == "Iran"


def test_when_livematch_away_team_is_usa_then_crosswalk_normalizes_to_united_states() -> None:
    """Crosswalk: "USA" (football-data.org alias) → "United States" (canonical)."""
    raw = {**_RAW_MATCH_FINISHED, "awayTeam": {"id": 2, "name": "USA"}}
    m = LiveMatch.model_validate(raw)
    assert m.away_team == "United States"


def test_when_livematch_home_team_is_turkiye_then_crosswalk_normalizes_to_turkey() -> None:
    """Crosswalk: "Türkiye" (football-data.org alias) → "Turkey" (canonical)."""
    raw = {**_RAW_MATCH_FINISHED, "homeTeam": {"id": 3, "name": "Türkiye"}}
    m = LiveMatch.model_validate(raw)
    assert m.home_team == "Turkey"


def test_when_livematch_home_team_is_korea_republic_then_crosswalk_normalizes_to_south_korea() -> (
    None
):
    """Crosswalk: "Korea Republic" (football-data.org alias) → "South Korea" (canonical)."""
    raw = {**_RAW_MATCH_FINISHED, "homeTeam": {"id": 4, "name": "Korea Republic"}}
    m = LiveMatch.model_validate(raw)
    assert m.home_team == "South Korea"


# ============================================================================
# AC-1 — TableRow: flatten nested team.name, ignore extra fields
# ============================================================================


def test_when_tablerow_validated_then_team_name_is_flat_field() -> None:
    row = TableRow.model_validate(_RAW_TABLE_ROW)
    assert row.team_name == "Germany"


def test_when_tablerow_validated_then_points_are_accessible() -> None:
    row = TableRow.model_validate(_RAW_TABLE_ROW)
    assert row.points == 6


def test_when_tablerow_validated_then_goals_for_is_accessible() -> None:
    row = TableRow.model_validate(_RAW_TABLE_ROW)
    assert row.goals_for == 5


def test_when_tablerow_validated_then_goal_difference_is_accessible() -> None:
    row = TableRow.model_validate(_RAW_TABLE_ROW)
    assert row.goal_difference == 4


def test_when_tablerow_validated_with_extra_field_then_it_is_silently_ignored() -> None:
    raw = {**_RAW_TABLE_ROW, "extraAttribute": "dropped"}
    row = TableRow.model_validate(raw)
    assert not hasattr(row, "extraAttribute")


# ============================================================================
# AC-1 — GroupStanding: group label accessible, table rows parsed
# ============================================================================


def test_when_groupstanding_validated_then_group_label_is_accessible() -> None:
    gs = GroupStanding.model_validate(_RAW_GROUP_STANDING)
    assert gs.group == "GROUP_A"


def test_when_groupstanding_validated_then_table_has_one_row() -> None:
    gs = GroupStanding.model_validate(_RAW_GROUP_STANDING)
    assert len(gs.table) == 1


def test_when_groupstanding_validated_then_table_row_team_name_is_correct() -> None:
    gs = GroupStanding.model_validate(_RAW_GROUP_STANDING)
    assert gs.table[0].team_name == "Germany"


# ============================================================================
# AC-1 — TournamentState: container with played / remaining / standings
# ============================================================================


def test_when_tournament_state_constructed_then_played_list_is_accessible() -> None:
    m = LiveMatch.model_validate(_RAW_MATCH_FINISHED)
    state = TournamentState(played=[m], remaining=[], standings=[])
    assert len(state.played) == 1


def test_when_tournament_state_constructed_then_remaining_list_is_accessible() -> None:
    m = LiveMatch.model_validate(_RAW_MATCH_SCHEDULED)
    state = TournamentState(played=[], remaining=[m], standings=[])
    assert len(state.remaining) == 1


def test_when_tournament_state_constructed_then_standings_list_is_accessible() -> None:
    gs = GroupStanding.model_validate(_RAW_GROUP_STANDING)
    state = TournamentState(played=[], remaining=[], standings=[gs])
    assert len(state.standings) == 1


# ============================================================================
# AC-2 — LiveTournamentAdapter.tournament_state(): split on status == "FINISHED"
#
# Interface assumption: the adapter calls client.get_competition_matches() and
# client.get_competition_standings().  These return dicts matching the v4 API
# shapes from requirements.md.  Adjust method names if the real FootballClient
# uses a different surface.
# ============================================================================


def _mock_client(matches: list, standings: list) -> MagicMock:
    """
    Duck-typed stub for FootballClient.  Returns raw v4-shape dicts so the
    adapter can parse them identically to the live API.
    Uses client.get() with side_effect to match the adapter's actual call pattern
    (matches endpoint first, standings endpoint second).
    """
    client = MagicMock()
    client.get.side_effect = [{"matches": matches}, {"standings": standings}]
    return client


def test_when_adapter_receives_finished_match_then_it_appears_in_played() -> None:
    adapter = LiveTournamentAdapter(
        _mock_client([_RAW_MATCH_FINISHED, _RAW_MATCH_SCHEDULED], [_RAW_GROUP_STANDING])
    )
    state = adapter.tournament_state()
    assert len(state.played) == 1
    assert state.played[0].status == "FINISHED"


def test_when_adapter_receives_scheduled_match_then_it_appears_in_remaining() -> None:
    adapter = LiveTournamentAdapter(
        _mock_client([_RAW_MATCH_FINISHED, _RAW_MATCH_SCHEDULED], [_RAW_GROUP_STANDING])
    )
    state = adapter.tournament_state()
    assert len(state.remaining) == 1
    assert state.remaining[0].status != "FINISHED"


def test_when_adapter_returns_state_then_standings_list_is_populated() -> None:
    adapter = LiveTournamentAdapter(_mock_client([_RAW_MATCH_FINISHED], [_RAW_GROUP_STANDING]))
    state = adapter.tournament_state()
    assert len(state.standings) >= 1


def test_when_adapter_has_only_knockout_matches_then_played_and_remaining_are_empty() -> None:
    """Only GROUP_STAGE fixtures appear in state; LAST_32 slots are excluded."""
    knockout = {**_RAW_MATCH_FINISHED, "stage": "LAST_32", "group": None}
    adapter = LiveTournamentAdapter(_mock_client([knockout], []))
    state = adapter.tournament_state()
    assert len(state.played) == 0
    assert len(state.remaining) == 0


def test_when_adapter_has_paused_match_then_it_appears_in_remaining_not_played() -> None:
    """PAUSED is not FINISHED — must go into remaining."""
    paused = {**_RAW_MATCH_FINISHED, "id": 9999, "status": "PAUSED"}
    adapter = LiveTournamentAdapter(_mock_client([paused], []))
    state = adapter.tournament_state()
    assert len(state.played) == 0
    assert len(state.remaining) == 1


def test_when_adapter_has_multiple_finished_matches_then_all_appear_in_played() -> None:
    second_finished = {
        **_RAW_MATCH_FINISHED,
        "id": 1003,
        "homeTeam": {"id": 800, "name": "France"},
        "awayTeam": {"id": 801, "name": "Argentina"},
    }
    adapter = LiveTournamentAdapter(
        _mock_client([_RAW_MATCH_FINISHED, second_finished], [_RAW_GROUP_STANDING])
    )
    state = adapter.tournament_state()
    assert len(state.played) == 2
    assert all(m.status == "FINISHED" for m in state.played)


# ============================================================================
# AC-3 — fetch_tournament_state(client=None) constructs a default FootballClient
# ============================================================================


def test_when_fetch_tournament_state_called_with_none_then_football_client_is_constructed(
    monkeypatch,
) -> None:
    """
    Criterion: constructs a default FootballClient when none is passed.
    We patch the FootballClient name in the live module's namespace and verify
    exactly one instance is created.
    """
    import worldcup_playoff.data.live as live_module

    constructed: list = []

    def _capturing_factory(*args, **kwargs):
        obj = MagicMock()
        constructed.append(obj)
        return obj

    monkeypatch.setattr(live_module, "FootballClient", _capturing_factory)
    fetch_tournament_state(client=None)
    assert len(constructed) == 1, (
        "Expected exactly one FootballClient to be constructed when client=None"
    )


def test_when_fetch_tournament_state_called_with_explicit_client_then_no_new_client_is_built(
    monkeypatch,
) -> None:
    """When a client is supplied, no additional FootballClient must be created."""
    import worldcup_playoff.data.live as live_module

    constructed: list = []

    def _capturing_factory(*args, **kwargs):
        obj = MagicMock()
        constructed.append(obj)
        return obj

    monkeypatch.setattr(live_module, "FootballClient", _capturing_factory)
    fetch_tournament_state(client=_mock_client([], []))
    assert len(constructed) == 0


def test_when_fetch_tournament_state_called_without_api_key_then_no_exception_is_raised(
    monkeypatch,
) -> None:
    """Key is optional — criterion says 'key optional'."""
    import worldcup_playoff.data.live as live_module

    monkeypatch.setattr(live_module, "FootballClient", lambda *a, **kw: _mock_client([], []))
    # Must not raise even when no key is configured
    result = fetch_tournament_state(client=None)
    assert isinstance(result, TournamentState)


# ============================================================================
# AC-4 — build_state_from_results(df): offline WC2026 group state from martj42
# ============================================================================

# Minimal complete group: four teams, round-robin (6 fixtures).
# First 4 rows are played (non-None scores); last 2 are unplayed.
_GROUP_ALPHA = [
    _wc2026_row("Alpha", "Beta", 2, 1, played=True),
    _wc2026_row("Gamma", "Delta", 1, 1, played=True),
    _wc2026_row("Alpha", "Gamma", 3, 0, played=True),
    _wc2026_row("Beta", "Delta", 0, 2, played=True),
    _wc2026_row("Alpha", "Delta", 0, 0, played=False),
    _wc2026_row("Beta", "Gamma", 0, 0, played=False),
]

# A disjoint second group (different teams, no overlap with _GROUP_ALPHA)
_GROUP_BRAVO = [
    _wc2026_row("Echo", "Foxtrot", 1, 0, played=True),
    _wc2026_row("Golf", "Hotel", 2, 2, played=True),
    _wc2026_row("Echo", "Golf", 0, 1, played=True),
    _wc2026_row("Foxtrot", "Hotel", 3, 0, played=True),
    _wc2026_row("Echo", "Hotel", 0, 0, played=False),
    _wc2026_row("Foxtrot", "Golf", 0, 0, played=False),
]

_NON_WC_ROW = ("2025-03-22", "Alpha", "Omega", 1.0, 0.0, "Friendly", "London", "England", False)
_WC_QUAL_ROW = (
    "2025-11-15",
    "Alpha",
    "Zeta",
    2.0,
    0.0,
    "FIFA World Cup qualification",
    "NY",
    "United States",
    False,
)


def test_when_build_state_from_results_given_wc2026_rows_then_played_have_non_null_scores() -> None:
    state = build_state_from_results(_df(*_GROUP_ALPHA))
    assert len(state.played) > 0
    for m in state.played:
        assert m.home_score is not None
        assert m.away_score is not None


def test_when_build_state_from_results_given_wc2026_rows_then_remaining_have_null_scores() -> None:
    state = build_state_from_results(_df(*_GROUP_ALPHA))
    assert len(state.remaining) > 0
    for m in state.remaining:
        assert m.home_score is None
        assert m.away_score is None


def test_when_build_state_from_results_receives_non_wc_row_then_it_is_excluded() -> None:
    state = build_state_from_results(_df(*_GROUP_ALPHA, _NON_WC_ROW))
    all_home_teams = {m.home_team for m in state.played} | {m.home_team for m in state.remaining}
    assert "Omega" not in all_home_teams


def test_when_build_state_from_results_receives_wc_qualification_row_then_it_is_excluded() -> None:
    state = build_state_from_results(_df(*_GROUP_ALPHA, _WC_QUAL_ROW))
    all_home_teams = {m.home_team for m in state.played} | {m.home_team for m in state.remaining}
    assert "Zeta" not in all_home_teams


def test_when_build_state_from_results_receives_only_non_wc2026_data_then_state_is_empty() -> None:
    state = build_state_from_results(_df(_NON_WC_ROW, _WC_QUAL_ROW))
    assert len(state.played) == 0
    assert len(state.remaining) == 0


def test_when_four_teams_play_full_round_robin_then_one_group_is_recovered_via_union_find() -> None:
    """
    Criterion: 'recovers groups via union-find'.
    Six fixtures among four mutually-connected teams form exactly one component.
    """
    state = build_state_from_results(_df(*_GROUP_ALPHA))
    assert len(state.standings) == 1


def test_when_two_disjoint_groups_play_then_two_groups_are_recovered_via_union_find() -> None:
    """Union-find must produce two separate components for two disjoint groups."""
    state = build_state_from_results(_df(*_GROUP_ALPHA, *_GROUP_BRAVO))
    assert len(state.standings) == 2


def test_when_build_state_from_results_given_four_played_fixtures_then_four_are_in_played() -> None:
    """Exact count check: 4 played rows → len(state.played) == 4."""
    state = build_state_from_results(_df(*_GROUP_ALPHA))
    assert len(state.played) == 4


def test_when_build_state_from_results_given_two_remaining_fixtures_then_two_are_in_remaining() -> (
    None
):
    """Exact count check: 2 unplayed rows → len(state.remaining) == 2."""
    state = build_state_from_results(_df(*_GROUP_ALPHA))
    assert len(state.remaining) == 2


# --- Property: played + remaining == total WC2026 fixture count ----------------


@given(played_count=st.integers(min_value=0, max_value=6))
@settings(max_examples=20)
def test_when_build_state_from_results_then_played_plus_remaining_equals_total_wc2026_count(
    played_count: int,
) -> None:
    """
    Invariant: len(played) + len(remaining) == number of WC2026 fixtures in df.
    Holds for every possible played/unplayed split of a 6-fixture round-robin.
    """
    pairs = [
        ("P1", "P2"),
        ("P3", "P4"),
        ("P1", "P3"),
        ("P2", "P4"),
        ("P1", "P4"),
        ("P2", "P3"),
    ]
    rows = [_wc2026_row(h, a, 1, 0, played=(i < played_count)) for i, (h, a) in enumerate(pairs)]
    state = build_state_from_results(_df(*rows))
    assert len(state.played) + len(state.remaining) == len(pairs)


# ============================================================================
# AC-5 — live_fixtures_to_df(state): martj42 schema, None-team slots dropped
# ============================================================================


def _minimal_state(*, played=None, remaining=None, standings=None) -> TournamentState:
    return TournamentState(
        played=played or [],
        remaining=remaining or [],
        standings=standings or [],
    )


def test_when_live_fixtures_to_df_called_then_result_is_a_dataframe() -> None:
    m = LiveMatch.model_validate(_RAW_MATCH_FINISHED)
    df = live_fixtures_to_df(_minimal_state(played=[m]))
    assert isinstance(df, pd.DataFrame)


def test_when_live_fixtures_to_df_called_then_result_has_martj42_schema_columns() -> None:
    m = LiveMatch.model_validate(_RAW_MATCH_FINISHED)
    df = live_fixtures_to_df(_minimal_state(played=[m]))
    assert _MARTJ42_SCHEMA.issubset(set(df.columns))


def test_when_live_fixtures_to_df_called_with_empty_state_then_empty_dataframe_is_returned() -> (
    None
):
    df = live_fixtures_to_df(_minimal_state())
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_when_live_fixtures_to_df_has_resolved_match_then_one_row_is_produced() -> None:
    m = LiveMatch.model_validate(_RAW_MATCH_FINISHED)
    df = live_fixtures_to_df(_minimal_state(played=[m]))
    assert len(df) == 1


def test_when_live_fixtures_to_df_has_none_team_slot_then_that_row_is_dropped() -> None:
    """
    Criterion: 'dropping unresolved None-team placeholder slots'.
    Knockout slots have null homeTeam/awayTeam until groups resolve.
    """
    null_slot_raw = {
        **_RAW_MATCH_SCHEDULED,
        "id": 9000,
        "homeTeam": {"id": None, "name": None},
        "awayTeam": {"id": None, "name": None},
    }
    unresolved = LiveMatch.model_validate(null_slot_raw)
    resolved = LiveMatch.model_validate(_RAW_MATCH_FINISHED)
    df = live_fixtures_to_df(_minimal_state(played=[resolved], remaining=[unresolved]))
    # Only the resolved match must appear
    assert len(df) == 1
    assert df["home_team"].notna().all()
    assert df["away_team"].notna().all()


def test_when_live_fixtures_to_df_includes_both_played_and_remaining_then_all_resolved_appear() -> (
    None
):
    """Both played and remaining resolved fixtures must be present in the output."""
    played = LiveMatch.model_validate(_RAW_MATCH_FINISHED)
    remaining = LiveMatch.model_validate(_RAW_MATCH_SCHEDULED)
    df = live_fixtures_to_df(_minimal_state(played=[played], remaining=[remaining]))
    assert len(df) == 2


def test_when_live_fixtures_to_df_called_then_played_row_has_non_null_scores() -> None:
    m = LiveMatch.model_validate(_RAW_MATCH_FINISHED)
    df = live_fixtures_to_df(_minimal_state(played=[m]))
    assert pd.notna(df["home_score"].iloc[0])
    assert pd.notna(df["away_score"].iloc[0])


def test_when_live_fixtures_to_df_called_with_remaining_match_then_scores_are_null() -> None:
    m = LiveMatch.model_validate(_RAW_MATCH_SCHEDULED)
    df = live_fixtures_to_df(_minimal_state(remaining=[m]))
    assert pd.isna(df["home_score"].iloc[0])
    assert pd.isna(df["away_score"].iloc[0])


# --- Property: resolved matches never produce null team names ----------------


@given(
    match_count=st.integers(min_value=1, max_value=8),
)
@settings(max_examples=20)
def test_when_live_fixtures_to_df_given_resolved_matches_then_no_null_team_names(
    match_count: int,
) -> None:
    """
    Invariant: for any number of resolved (non-None team) fixtures,
    the output DataFrame contains no null values in home_team / away_team.
    """
    played = [
        LiveMatch.model_validate(
            {
                **_RAW_MATCH_FINISHED,
                "id": i,
                "homeTeam": {"id": 1, "name": "Germany"},
                "awayTeam": {"id": 2, "name": "Japan"},
            }
        )
        for i in range(match_count)
    ]
    df = live_fixtures_to_df(_minimal_state(played=played))
    assert df["home_team"].notna().all()
    assert df["away_team"].notna().all()


# ============================================================================
# Issue #28 comment — group-stage guard: knockout matches must not corrupt
# the union-find group recovery in build_state_from_results.
# ============================================================================


def test_when_wc2026_has_knockout_match_connecting_groups_then_both_groups_still_recovered():
    """
    Group-stage guard: a knockout fixture dated >= 2026-07-04 connecting teams
    from different groups must be excluded from union-find.
    Both 4-team groups must still be recovered intact.
    """
    knockout_row = (
        "2026-07-04",
        "Alpha",
        "Echo",
        2.0,
        1.0,
        "FIFA World Cup",
        "Dallas",
        "United States",
        True,
    )
    df = _df(*_GROUP_ALPHA, *_GROUP_BRAVO, knockout_row)
    state = build_state_from_results(df)
    assert len(state.standings) == 2


def test_when_wc2026_has_twelve_groups_plus_r32_knockout_then_exactly_twelve_groups_recovered():
    """Issue #28 comment: with 12 groups (72 fixtures) + 16 R32 knockout matches
    dated >= 2026-07-04, exactly twelve 4-team components must be recovered."""
    teams = [f"T{i:02d}" for i in range(48)]
    group_rows = [
        _wc2026_row(teams[g * 4 + i], teams[g * 4 + j], 1, 0)
        for g in range(12)
        for i in range(4)
        for j in range(i + 1, 4)
    ]
    # Knockout matches cross group boundaries; dated >= 2026-07-04 so excluded by guard
    knockout_rows = [
        (
            "2026-07-04",
            teams[k % 48],
            teams[(k + 4) % 48],
            1.0,
            0.0,
            "FIFA World Cup",
            "Dallas",
            "United States",
            True,
        )
        for k in range(16)
    ]
    state = build_state_from_results(_df(*group_rows, *knockout_rows))
    assert len(state.standings) == 12
