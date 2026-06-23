"""
Source-blind example tests for issue #3:
  feat: live WC2026 adapter over FootballClient (matches, standings, tournament state)

Every test is derived solely from the acceptance criteria and the football-data.org v4
data contract in requirements.md.  No implementation source was read; these tests
describe the contract the implementation must satisfy (Red phase of TDD).

Choices made where the spec is silent:
  - `fetch_matches()` returns ALL matches from the API (not only group-stage);
    partitioning into played / remaining happens inside `tournament_state()`.
  - `normalize_team` lives in `worldcup_playoff.data.crosswalk` (the module is listed
    in the git tree as an untracked file committed alongside this issue).
  - The `live` field on `AppConfig` has a default so `AppConfig` can be inspected
    without a TOML file; if it is required, the test checks `model_fields` instead.
"""

import typing
from unittest.mock import MagicMock, patch

from hypothesis import given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Shape-accurate fixtures derived from the football-data.org v4 data contract
# (requirements.md §Data Contracts)
# ---------------------------------------------------------------------------

_FINISHED_GROUP_MATCH = {
    "id": 101,
    "utcDate": "2026-06-15T15:00:00Z",
    "status": "FINISHED",
    "stage": "GROUP_STAGE",
    "group": "GROUP_A",
    "matchday": 1,
    "homeTeam": {"id": 10, "name": "Brazil"},
    "awayTeam": {"id": 11, "name": "Argentina"},
    "score": {
        "winner": "HOME_TEAM",
        "duration": "REGULAR",
        "fullTime": {"home": 2, "away": 1},
        "halfTime": {"home": 1, "away": 0},
    },
}

_SCHEDULED_GROUP_MATCH = {
    "id": 102,
    "utcDate": "2026-06-25T18:00:00Z",
    "status": "SCHEDULED",
    "stage": "GROUP_STAGE",
    "group": "GROUP_A",
    "matchday": 3,
    "homeTeam": {"id": 10, "name": "Brazil"},
    "awayTeam": {"id": 12, "name": "France"},
    "score": {
        "winner": None,
        "duration": "REGULAR",
        "fullTime": {"home": None, "away": None},
        "halfTime": {"home": None, "away": None},
    },
}

# LAST_32 slot not yet filled → team names are null
_LAST_32_NULL_TEAMS_MATCH = {
    "id": 103,
    "utcDate": "2026-07-01T20:00:00Z",
    "status": "SCHEDULED",
    "stage": "LAST_32",
    "group": None,
    "matchday": None,
    "homeTeam": {"id": None, "name": None},
    "awayTeam": {"id": None, "name": None},
    "score": {
        "winner": None,
        "duration": "REGULAR",
        "fullTime": {"home": None, "away": None},
        "halfTime": {"home": None, "away": None},
    },
}

_MATCHES_JSON = {
    "filters": {},
    "resultSet": {"count": 3, "first": "2026-06-15", "last": "2026-07-01", "played": 1},
    "competition": {"id": 2000, "name": "FIFA World Cup"},
    "matches": [
        _FINISHED_GROUP_MATCH,
        _SCHEDULED_GROUP_MATCH,
        _LAST_32_NULL_TEAMS_MATCH,
    ],
}


def _make_standings_json(n_groups: int = 12) -> dict:
    """Return a standings payload with *n_groups* groups (default 12 for WC2026)."""
    groups = [f"GROUP_{chr(65 + i)}" for i in range(n_groups)]
    return {
        "standings": [
            {
                "stage": "GROUP_STAGE",
                "type": "TOTAL",
                "group": group,
                "table": [
                    {
                        "position": 1,
                        "team": {"id": idx * 4 + 1, "name": f"Team_{group}_1"},
                        "playedGames": 1,
                        "form": "W",
                        "won": 1,
                        "draw": 0,
                        "lost": 0,
                        "points": 3,
                        "goalsFor": 2,
                        "goalsAgainst": 0,
                        "goalDifference": 2,
                    }
                ],
            }
            for idx, group in enumerate(groups)
        ]
    }


_STANDINGS_JSON = _make_standings_json(12)


def _client_for_tournament_state() -> MagicMock:
    """Mock whose .get() yields matches JSON then standings JSON (two calls)."""
    client = MagicMock()
    client.get.side_effect = [_MATCHES_JSON, _STANDINGS_JSON]
    return client


def _client_returning_matches() -> MagicMock:
    client = MagicMock()
    client.get.return_value = _MATCHES_JSON
    return client


def _client_returning_standings() -> MagicMock:
    client = MagicMock()
    client.get.return_value = _STANDINGS_JSON
    return client


# ---------------------------------------------------------------------------
# AC1 — LiveTournamentAdapter interface
# ---------------------------------------------------------------------------


class TestLiveTournamentAdapterInterface:
    def test_when_adapter_constructed_then_instance_is_created(self):
        from worldcup_playoff.data.live import LiveTournamentAdapter

        adapter = LiveTournamentAdapter(_client_returning_matches(), competition="WC")
        assert adapter is not None

    def test_when_fetch_matches_called_then_a_list_is_returned(self):
        from worldcup_playoff.data.live import LiveTournamentAdapter

        adapter = LiveTournamentAdapter(_client_returning_matches(), competition="WC")
        result = adapter.fetch_matches()
        assert isinstance(result, list)

    def test_when_fetch_matches_called_then_every_item_is_a_live_match(self):
        from worldcup_playoff.data.live import LiveMatch, LiveTournamentAdapter

        adapter = LiveTournamentAdapter(_client_returning_matches(), competition="WC")
        result = adapter.fetch_matches()
        assert all(isinstance(m, LiveMatch) for m in result)

    def test_when_fetch_standings_called_then_a_list_is_returned(self):
        from worldcup_playoff.data.live import LiveTournamentAdapter

        adapter = LiveTournamentAdapter(_client_returning_standings(), competition="WC")
        result = adapter.fetch_standings()
        assert isinstance(result, list)

    def test_when_fetch_standings_called_then_every_item_is_a_group_standing(self):
        from worldcup_playoff.data.live import GroupStanding, LiveTournamentAdapter

        adapter = LiveTournamentAdapter(_client_returning_standings(), competition="WC")
        result = adapter.fetch_standings()
        assert all(isinstance(s, GroupStanding) for s in result)

    def test_when_tournament_state_called_then_a_tournament_state_is_returned(self):
        from worldcup_playoff.data.live import LiveTournamentAdapter, TournamentState

        adapter = LiveTournamentAdapter(_client_for_tournament_state(), competition="WC")
        result = adapter.tournament_state()
        assert isinstance(result, TournamentState)


# ---------------------------------------------------------------------------
# AC2 — tournament_state() partitioning (played / remaining / standings)
# ---------------------------------------------------------------------------


class TestTournamentStatePartitioning:
    def test_when_tournament_state_called_then_played_is_a_list(self):
        from worldcup_playoff.data.live import LiveTournamentAdapter

        state = LiveTournamentAdapter(
            _client_for_tournament_state(), competition="WC"
        ).tournament_state()
        assert isinstance(state.played, list)

    def test_when_tournament_state_called_then_remaining_group_fixtures_is_a_list(self):
        from worldcup_playoff.data.live import LiveTournamentAdapter

        state = LiveTournamentAdapter(
            _client_for_tournament_state(), competition="WC"
        ).tournament_state()
        assert isinstance(state.remaining_group_fixtures, list)

    def test_when_finished_group_match_present_then_it_is_in_played(self):
        """FINISHED group-stage match (id=101) must land in TournamentState.played."""
        from worldcup_playoff.data.live import LiveTournamentAdapter

        state = LiveTournamentAdapter(
            _client_for_tournament_state(), competition="WC"
        ).tournament_state()
        played_ids = {m.id for m in state.played}
        assert 101 in played_ids

    def test_when_scheduled_group_match_present_then_it_is_in_remaining(self):
        """SCHEDULED group-stage match (id=102) must land in remaining_group_fixtures."""
        from worldcup_playoff.data.live import LiveTournamentAdapter

        state = LiveTournamentAdapter(
            _client_for_tournament_state(), competition="WC"
        ).tournament_state()
        remaining_ids = {m.id for m in state.remaining_group_fixtures}
        assert 102 in remaining_ids

    def test_when_tournament_state_partitions_matches_then_played_and_remaining_are_disjoint(self):
        from worldcup_playoff.data.live import LiveTournamentAdapter

        state = LiveTournamentAdapter(
            _client_for_tournament_state(), competition="WC"
        ).tournament_state()
        played_ids = {m.id for m in state.played}
        remaining_ids = {m.id for m in state.remaining_group_fixtures}
        assert played_ids.isdisjoint(remaining_ids)

    def test_when_tournament_state_called_then_knockout_match_is_not_in_played(self):
        """LAST_32 match (id=103) is not a group-stage match → must not appear in played."""
        from worldcup_playoff.data.live import LiveTournamentAdapter

        state = LiveTournamentAdapter(
            _client_for_tournament_state(), competition="WC"
        ).tournament_state()
        played_ids = {m.id for m in state.played}
        assert 103 not in played_ids

    def test_when_tournament_state_called_then_knockout_match_is_not_in_remaining(self):
        """LAST_32 match (id=103) must not appear in remaining_group_fixtures either."""
        from worldcup_playoff.data.live import LiveTournamentAdapter

        state = LiveTournamentAdapter(
            _client_for_tournament_state(), competition="WC"
        ).tournament_state()
        remaining_ids = {m.id for m in state.remaining_group_fixtures}
        assert 103 not in remaining_ids

    def test_when_tournament_state_called_then_standings_carries_twelve_groups(self):
        """WC2026 has 12 groups — all must be present in TournamentState.standings."""
        from worldcup_playoff.data.live import LiveTournamentAdapter

        state = LiveTournamentAdapter(
            _client_for_tournament_state(), competition="WC"
        ).tournament_state()
        assert len(state.standings) == 12

    def test_when_tournament_state_called_then_standings_is_a_list(self):
        from worldcup_playoff.data.live import LiveTournamentAdapter

        state = LiveTournamentAdapter(
            _client_for_tournament_state(), competition="WC"
        ).tournament_state()
        assert isinstance(state.standings, list)


# ---------------------------------------------------------------------------
# AC3 — null team names → None; non-null names pass through normalize_team
# ---------------------------------------------------------------------------


class TestNullTeamParsing:
    def test_when_knockout_match_has_null_home_team_name_then_home_team_is_none(self):
        from worldcup_playoff.data.live import LiveTournamentAdapter

        matches = LiveTournamentAdapter(
            _client_returning_matches(), competition="WC"
        ).fetch_matches()
        knockout = next(m for m in matches if m.id == 103)
        assert knockout.home_team is None

    def test_when_knockout_match_has_null_away_team_name_then_away_team_is_none(self):
        from worldcup_playoff.data.live import LiveTournamentAdapter

        matches = LiveTournamentAdapter(
            _client_returning_matches(), competition="WC"
        ).fetch_matches()
        knockout = next(m for m in matches if m.id == 103)
        assert knockout.away_team is None

    def test_when_group_match_has_non_null_home_team_then_home_team_is_a_string(self):
        from worldcup_playoff.data.live import LiveTournamentAdapter

        matches = LiveTournamentAdapter(
            _client_returning_matches(), competition="WC"
        ).fetch_matches()
        group_match = next(m for m in matches if m.id == 101)
        assert isinstance(group_match.home_team, str)

    def test_when_group_match_has_non_null_away_team_then_away_team_is_a_string(self):
        from worldcup_playoff.data.live import LiveTournamentAdapter

        matches = LiveTournamentAdapter(
            _client_returning_matches(), competition="WC"
        ).fetch_matches()
        group_match = next(m for m in matches if m.id == 101)
        assert isinstance(group_match.away_team, str)

    def test_when_adapter_processes_null_team_names_then_no_exception_is_raised(self):
        """Parsing a LAST_32 match with both team names null must never crash."""
        from worldcup_playoff.data.live import LiveTournamentAdapter

        # simply calling fetch_matches on a payload that contains null-team slots must succeed
        LiveTournamentAdapter(_client_returning_matches(), competition="WC").fetch_matches()


# Property: normalize_team is a total function on any non-empty string (never raises)
@given(st.text(min_size=1))
def test_when_any_non_empty_string_passed_to_normalize_team_then_no_error_is_raised(name: str):
    """Invariant from AC3: normalize_team must accept any non-empty name without raising."""
    from worldcup_playoff.data.crosswalk import normalize_team  # noqa: PLC0415

    normalize_team(name)  # must not raise


# ---------------------------------------------------------------------------
# AC4 — Pydantic value objects respect extra="ignore"
# ---------------------------------------------------------------------------


class TestPydanticExtraIgnore:
    def test_when_match_json_has_extra_fields_then_live_match_is_still_returned(self):
        """extra='ignore' → unknown fields in the API payload must not raise ValidationError."""
        match_with_extra = {**_FINISHED_GROUP_MATCH, "unknownExtraField": "surprise"}
        matches_json_with_extra = {**_MATCHES_JSON, "matches": [match_with_extra]}
        client = MagicMock()
        client.get.return_value = matches_json_with_extra

        from worldcup_playoff.data.live import LiveTournamentAdapter

        result = LiveTournamentAdapter(client, competition="WC").fetch_matches()
        assert len(result) == 1

    def test_when_standings_json_has_extra_fields_then_group_standing_is_still_returned(self):
        standing_with_extra = {
            **_STANDINGS_JSON["standings"][0],
            "unexpectedKey": "ignored_value",
        }
        standings_with_extra = {"standings": [standing_with_extra]}
        client = MagicMock()
        client.get.return_value = standings_with_extra

        from worldcup_playoff.data.live import LiveTournamentAdapter

        result = LiveTournamentAdapter(client, competition="WC").fetch_standings()
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_when_tournament_state_payload_has_extra_fields_then_tournament_state_is_returned(self):
        match_with_extra = {**_FINISHED_GROUP_MATCH, "bogusField": 99}
        matches_json_with_extra = {**_MATCHES_JSON, "matches": [match_with_extra]}
        client = MagicMock()
        client.get.side_effect = [matches_json_with_extra, _STANDINGS_JSON]

        from worldcup_playoff.data.live import LiveTournamentAdapter

        state = LiveTournamentAdapter(client, competition="WC").tournament_state()
        assert state is not None


# ---------------------------------------------------------------------------
# AC5 — module-level fetch_tournament_state default-constructs FootballClient
# ---------------------------------------------------------------------------


class TestFetchTournamentStateFunction:
    def test_when_called_without_client_then_football_client_is_constructed(self):
        from worldcup_playoff.data.live import fetch_tournament_state

        with patch("worldcup_playoff.data.live.FootballClient") as MockFC:
            instance = MagicMock()
            instance.get.side_effect = [_MATCHES_JSON, _STANDINGS_JSON]
            MockFC.return_value = instance
            fetch_tournament_state()
            MockFC.assert_called_once()

    def test_when_called_with_explicit_client_then_football_client_is_not_constructed(self):
        from worldcup_playoff.data.live import fetch_tournament_state

        client = _client_for_tournament_state()
        with patch("worldcup_playoff.data.live.FootballClient") as MockFC:
            fetch_tournament_state(client=client)
            MockFC.assert_not_called()

    def test_when_called_without_client_then_tournament_state_is_returned(self):
        from worldcup_playoff.data.live import TournamentState, fetch_tournament_state

        with patch("worldcup_playoff.data.live.FootballClient") as MockFC:
            instance = MagicMock()
            instance.get.side_effect = [_MATCHES_JSON, _STANDINGS_JSON]
            MockFC.return_value = instance
            result = fetch_tournament_state()
        assert isinstance(result, TournamentState)

    def test_when_competition_parameter_default_is_wc(self):
        """The default value for the `competition` parameter must be 'WC'."""
        import inspect

        from worldcup_playoff.data.live import fetch_tournament_state

        sig = inspect.signature(fetch_tournament_state)
        assert sig.parameters["competition"].default == "WC"

    def test_when_client_parameter_default_is_none(self):
        """The default value for the `client` parameter must be None."""
        import inspect

        from worldcup_playoff.data.live import fetch_tournament_state

        sig = inspect.signature(fetch_tournament_state)
        assert sig.parameters["client"].default is None


# ---------------------------------------------------------------------------
# AC6 — LiveConfig on AppConfig + __all__ exports from worldcup_playoff.data
# ---------------------------------------------------------------------------


class TestLiveConfigAndPackageExports:
    # --- config.py ---

    def test_when_live_config_imported_from_config_then_it_exists(self):
        from worldcup_playoff.config import LiveConfig  # noqa: F401

    def test_when_app_config_model_fields_inspected_then_live_field_is_present(self):
        from worldcup_playoff.config import AppConfig

        assert "live" in AppConfig.model_fields

    def test_when_app_config_live_field_is_annotated_then_it_is_live_config_type(self):
        from worldcup_playoff.config import AppConfig, LiveConfig

        field = AppConfig.model_fields["live"]
        annotation = field.annotation
        # Unwrap Optional[LiveConfig] / Union[LiveConfig, None] if present
        origin = getattr(annotation, "__origin__", None)
        if origin is typing.Union:
            inner = [t for t in annotation.__args__ if t is not type(None)]
            assert any(t is LiveConfig for t in inner), (
                f"AppConfig.live annotation {annotation} does not include LiveConfig"
            )
        else:
            assert annotation is LiveConfig, (
                f"AppConfig.live annotation is {annotation}, expected LiveConfig"
            )

    # --- worldcup_playoff/data/__init__.py ---

    def test_when_live_tournament_adapter_imported_from_data_package_then_it_is_available(self):
        from worldcup_playoff.data import LiveTournamentAdapter  # noqa: F401

    def test_when_live_match_imported_from_data_package_then_it_is_available(self):
        from worldcup_playoff.data import LiveMatch  # noqa: F401

    def test_when_group_standing_imported_from_data_package_then_it_is_available(self):
        from worldcup_playoff.data import GroupStanding  # noqa: F401

    def test_when_tournament_state_imported_from_data_package_then_it_is_available(self):
        from worldcup_playoff.data import TournamentState  # noqa: F401

    def test_when_fetch_tournament_state_imported_from_data_package_then_it_is_available(self):
        from worldcup_playoff.data import fetch_tournament_state  # noqa: F401

    def test_when_data_package_all_inspected_then_live_tournament_adapter_is_listed(self):
        import worldcup_playoff.data as data_pkg

        assert "LiveTournamentAdapter" in data_pkg.__all__

    def test_when_data_package_all_inspected_then_live_match_is_listed(self):
        import worldcup_playoff.data as data_pkg

        assert "LiveMatch" in data_pkg.__all__

    def test_when_data_package_all_inspected_then_group_standing_is_listed(self):
        import worldcup_playoff.data as data_pkg

        assert "GroupStanding" in data_pkg.__all__

    def test_when_data_package_all_inspected_then_tournament_state_is_listed(self):
        import worldcup_playoff.data as data_pkg

        assert "TournamentState" in data_pkg.__all__

    def test_when_data_package_all_inspected_then_fetch_tournament_state_is_listed(self):
        import worldcup_playoff.data as data_pkg

        assert "fetch_tournament_state" in data_pkg.__all__


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------


def _make_group_match(match_id: int, status: str) -> dict:
    return {
        "id": match_id,
        "utcDate": "2026-06-20T12:00:00Z",
        "status": status,
        "stage": "GROUP_STAGE",
        "group": "GROUP_A",
        "matchday": 1,
        "homeTeam": {"id": 1, "name": "TeamA"},
        "awayTeam": {"id": 2, "name": "TeamB"},
        "score": {
            "winner": "HOME_TEAM" if status == "FINISHED" else None,
            "duration": "REGULAR",
            "fullTime": (
                {"home": 1, "away": 0} if status == "FINISHED" else {"home": None, "away": None}
            ),
            "halfTime": (
                {"home": 0, "away": 0} if status == "FINISHED" else {"home": None, "away": None}
            ),
        },
    }


@given(
    statuses=st.lists(
        st.sampled_from(["FINISHED", "SCHEDULED", "TIMED"]),
        min_size=0,
        max_size=8,
    )
)
@settings(max_examples=40)
def test_when_group_stage_matches_have_mixed_statuses_then_partition_covers_all_of_them(
    statuses: list,
) -> None:
    """
    Invariant from AC2: played ∪ remaining_group_fixtures must cover every group-stage
    match exactly once — no match may be silently dropped from TournamentState.
    """
    from worldcup_playoff.data.live import LiveTournamentAdapter

    group_matches = [_make_group_match(i, status) for i, status in enumerate(statuses)]
    matches_json = {**_MATCHES_JSON, "matches": group_matches}

    client = MagicMock()
    client.get.side_effect = [matches_json, _STANDINGS_JSON]

    state = LiveTournamentAdapter(client, competition="WC").tournament_state()
    total_partitioned = len(state.played) + len(state.remaining_group_fixtures)
    assert total_partitioned == len(group_matches)
