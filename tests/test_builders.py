"""Tests for the CSV dataset builders (teams, matches, ranking, players, match_details)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from worldcup_playoff.data.builders import (
    MatchDetailsBuilder,
    MatchesBuilder,
    PlayersBuilder,
    RankingBuilder,
    TeamsBuilder,
    _MATCH_DETAILS_COLUMNS,
    _MATCHES_COLUMNS,
    _PLAYERS_COLUMNS,
    _RANKING_COLUMNS,
    _TEAMS_COLUMNS,
    build_match_details_csv,
    build_matches_csv,
    build_players_csv,
    build_ranking_csv,
    build_teams_csv,
)


# ---------------------------------------------------------------------------
# Canned API responses
# ---------------------------------------------------------------------------


def _team_response(n: int = 3) -> dict:
    return {
        "teams": [
            {
                "id": 100 + i,
                "name": f"Team {i}",
                "shortName": f"T{i}",
                "tla": f"T0{i}",
                "area": {"name": f"Country{i}"},
            }
            for i in range(n)
        ]
    }


def _matches_response(code: str = "WC", year: int = 2022, n: int = 2) -> dict:
    return {
        "matches": [
            {
                "id": 1000 + i,
                "utcDate": f"{year}-07-0{i + 1}T18:00:00Z",
                "homeTeam": {"name": f"HomeTeam{i}", "shortName": f"HT{i}"},
                "awayTeam": {"name": f"AwayTeam{i}", "shortName": f"AT{i}"},
                "score": {
                    "fullTime": {"home": i + 1, "away": i},
                },
            }
            for i in range(n)
        ]
    }


def _standings_response(n_teams: int = 4) -> dict:
    return {
        "standings": [
            {
                "table": [
                    {
                        "position": i + 1,
                        "team": {"name": f"Team{i}"},
                        "playedGames": 3,
                        "won": 2,
                        "draw": 1,
                        "lost": 0,
                        "points": 7,
                        "goalsFor": 5,
                        "goalsAgainst": 2,
                    }
                    for i in range(n_teams)
                ]
            }
        ]
    }


def _players_teams_response(n: int = 2) -> dict:
    return {"teams": [{"id": 200 + i, "name": f"Nation{i}"} for i in range(n)]}


def _squad_response(n: int = 3) -> dict:
    return {
        "squad": [
            {
                "id": 300 + i,
                "name": f"Player {i}",
                "nationality": "Brazilian",
                "position": "Midfielder",
            }
            for i in range(n)
        ]
    }


def _match_detail_response(match_id: int = 1001, home_goals: int = 2, away_goals: int = 1) -> dict:
    return {
        "id": match_id,
        "score": {"fullTime": {"home": home_goals, "away": away_goals}},
        "statistics": [],
    }


# ---------------------------------------------------------------------------
# TeamsBuilder
# ---------------------------------------------------------------------------


class TestTeamsBuilder:
    def _mock_client(self, response: dict) -> MagicMock:
        client = MagicMock()
        client.get.return_value = response
        return client

    def test_build_produces_correct_columns(self) -> None:
        client = self._mock_client(_team_response(3))
        builder = TeamsBuilder(client, competition_codes=["WC"])
        df = builder.build()
        assert list(df.columns) == _TEAMS_COLUMNS

    def test_build_deduplicates_teams(self) -> None:
        """Same team returned by two competitions is only counted once."""
        client = MagicMock()
        client.get.return_value = _team_response(2)
        builder = TeamsBuilder(client, competition_codes=["WC", "EC"])
        df = builder.build()
        assert df["TEAM_ID"].nunique() == len(df)

    def test_build_raises_when_all_fail(self) -> None:
        client = MagicMock()
        client.get.side_effect = RuntimeError("network error")
        builder = TeamsBuilder(client, competition_codes=["WC"])
        with pytest.raises(RuntimeError, match="No teams fetched"):
            builder.build()

    def test_build_skips_failed_competition(self) -> None:
        """One competition fails but the other succeeds — builder continues."""
        client = MagicMock()
        client.get.side_effect = [RuntimeError("down"), _team_response(2)]
        builder = TeamsBuilder(client, competition_codes=["FAIL", "WC"])
        df = builder.build()
        assert len(df) == 2


class TestBuildTeamsCsv:
    def test_writes_csv(self, tmp_path: Path) -> None:
        client = MagicMock()
        client.get.return_value = _team_response(5)

        out = tmp_path / "teams.csv"
        df = build_teams_csv(out, client=client)

        assert out.exists()
        assert len(df) == 5
        assert list(df.columns) == _TEAMS_COLUMNS


# ---------------------------------------------------------------------------
# MatchesBuilder
# ---------------------------------------------------------------------------


class TestMatchesBuilder:
    def test_build_correct_columns(self) -> None:
        client = MagicMock()
        client.get.return_value = _matches_response()
        builder = MatchesBuilder(client, start_year=2022, end_year=2022, competition_codes=["WC"])
        df = builder.build()
        assert list(df.columns) == _MATCHES_COLUMNS

    def test_build_raises_when_all_fail(self) -> None:
        client = MagicMock()
        client.get.side_effect = RuntimeError("network error")
        builder = MatchesBuilder(client, start_year=2022, end_year=2022, competition_codes=["WC"])
        with pytest.raises(RuntimeError, match="All match fetches failed"):
            builder.build()

    def test_build_skips_empty_seasons(self) -> None:
        """Seasons that return no matches are skipped without error."""
        client = MagicMock()
        client.get.side_effect = [
            {"matches": []},  # 2021 returns nothing
            _matches_response(year=2022, n=2),  # 2022 succeeds
        ]
        builder = MatchesBuilder(client, start_year=2021, end_year=2022, competition_codes=["WC"])
        df = builder.build()
        assert len(df) == 2

    def test_build_deduplicates_match_ids(self) -> None:
        """Same MATCH_ID appearing under two competition codes is deduplicated."""
        shared_response = _matches_response(n=2)
        client = MagicMock()
        client.get.return_value = shared_response
        builder = MatchesBuilder(
            client, start_year=2022, end_year=2022, competition_codes=["WC", "EC"]
        )
        df = builder.build()
        assert df["MATCH_ID"].nunique() == len(df)

    def test_extract_match_row_maps_fields_correctly(self) -> None:
        match = {
            "id": 9999,
            "utcDate": "2022-12-18T18:00:00Z",
            "homeTeam": {"name": "Brazil", "shortName": "BRA"},
            "awayTeam": {"name": "France", "shortName": "FRA"},
            "score": {"fullTime": {"home": 2, "away": 1}},
        }
        row = MatchesBuilder._extract_match_row(match, "WC", 2022)
        assert row is not None
        assert row["MATCH_ID"] == 9999
        assert row["HOME_TEAM"] == "Brazil"
        assert row["AWAY_TEAM"] == "France"
        assert row["HOME_GOALS"] == 2
        assert row["AWAY_GOALS"] == 1
        assert row["SEASON"] == 2022

    def test_extract_match_row_returns_none_for_missing_goals(self) -> None:
        match = {
            "id": 9999,
            "utcDate": "2022-12-18T18:00:00Z",
            "homeTeam": {"name": "Brazil"},
            "awayTeam": {"name": "France"},
            "score": {"fullTime": {"home": None, "away": 1}},
        }
        assert MatchesBuilder._extract_match_row(match, "WC", 2022) is None


class TestBuildMatchesCsv:
    def test_writes_csv(self, tmp_path: Path) -> None:
        client = MagicMock()
        client.get.return_value = _matches_response(n=3)

        out = tmp_path / "matches.csv"
        df = build_matches_csv(out, client=client, start_year=2022, end_year=2022)

        assert out.exists()
        assert len(df) == 3
        assert list(df.columns) == _MATCHES_COLUMNS


# ---------------------------------------------------------------------------
# RankingBuilder
# ---------------------------------------------------------------------------


class TestRankingBuilder:
    def test_build_correct_columns(self) -> None:
        client = MagicMock()
        client.get.return_value = _standings_response(4)
        builder = RankingBuilder(client, start_year=2022, end_year=2022, competition_codes=["WC"])
        df = builder.build()
        assert list(df.columns) == _RANKING_COLUMNS

    def test_build_raises_when_all_fail(self) -> None:
        client = MagicMock()
        client.get.side_effect = RuntimeError("network error")
        builder = RankingBuilder(client, start_year=2022, end_year=2022, competition_codes=["WC"])
        with pytest.raises(RuntimeError, match="All standings fetches failed"):
            builder.build()

    def test_extract_ranking_row_maps_fields(self) -> None:
        entry = {
            "position": 1,
            "team": {"name": "Brazil"},
            "playedGames": 3,
            "won": 2,
            "draw": 1,
            "lost": 0,
            "points": 7,
            "goalsFor": 5,
            "goalsAgainst": 2,
        }
        row = RankingBuilder._extract_ranking_row(entry, "WC", 2022)
        assert row is not None
        assert row["TEAM"] == "Brazil"
        assert row["POSITION"] == 1
        assert row["POINTS"] == 7


class TestBuildRankingCsv:
    def test_writes_csv(self, tmp_path: Path) -> None:
        client = MagicMock()
        client.get.return_value = _standings_response(4)

        out = tmp_path / "ranking.csv"
        df = build_ranking_csv(out, client=client, start_year=2022, end_year=2022)

        assert out.exists()
        assert list(df.columns) == _RANKING_COLUMNS


# ---------------------------------------------------------------------------
# PlayersBuilder
# ---------------------------------------------------------------------------


class TestPlayersBuilder:
    def test_build_correct_columns(self) -> None:
        client = MagicMock()
        client.get.side_effect = [
            _players_teams_response(2),
            _squad_response(3),
            _squad_response(3),
        ]
        builder = PlayersBuilder(client, competition="WC")
        df = builder.build()
        assert list(df.columns) == _PLAYERS_COLUMNS

    def test_build_raises_when_no_teams(self) -> None:
        client = MagicMock()
        client.get.return_value = {"teams": []}
        builder = PlayersBuilder(client, competition="WC")
        with pytest.raises(RuntimeError, match="No teams found"):
            builder.build()

    def test_build_raises_when_no_players(self) -> None:
        client = MagicMock()
        # Teams fetched OK but squads return nothing
        client.get.side_effect = [
            _players_teams_response(1),
            {"squad": []},
        ]
        builder = PlayersBuilder(client, competition="WC")
        with pytest.raises(RuntimeError, match="No player data"):
            builder.build()

    def test_build_skips_failed_squad_fetch(self) -> None:
        """A failed squad fetch does not crash the build — it just returns no players
        for that team (other teams still contribute)."""
        client = MagicMock()
        client.get.side_effect = [
            _players_teams_response(2),
            RuntimeError("timeout"),
            _squad_response(2),
        ]
        builder = PlayersBuilder(client, competition="WC")
        df = builder.build()
        assert len(df) == 2


class TestBuildPlayersCsv:
    def test_writes_csv(self, tmp_path: Path) -> None:
        client = MagicMock()
        client.get.side_effect = [
            _players_teams_response(2),
            _squad_response(3),
            _squad_response(3),
        ]
        out = tmp_path / "players.csv"
        df = build_players_csv(out, client=client, competition="WC")
        assert out.exists()
        assert list(df.columns) == _PLAYERS_COLUMNS


# ---------------------------------------------------------------------------
# MatchDetailsBuilder
# ---------------------------------------------------------------------------


class TestMatchDetailsBuilder:
    def test_build_correct_columns(self) -> None:
        client = MagicMock()
        client.get.return_value = _match_detail_response(match_id=1001)
        builder = MatchDetailsBuilder(client, match_ids=[1001])
        df = builder.build()
        assert list(df.columns) == _MATCH_DETAILS_COLUMNS

    def test_build_raises_when_all_fail(self) -> None:
        client = MagicMock()
        client.get.side_effect = RuntimeError("network error")
        builder = MatchDetailsBuilder(client, match_ids=[9999])
        with pytest.raises(RuntimeError, match="No match details fetched"):
            builder.build()

    def test_build_skips_failed_match_ids(self) -> None:
        client = MagicMock()
        client.get.side_effect = [
            RuntimeError("timeout"),
            _match_detail_response(match_id=1002),
        ]
        builder = MatchDetailsBuilder(client, match_ids=[9999, 1002])
        df = builder.build()
        assert len(df) == 1
        assert int(df.iloc[0]["MATCH_ID"]) == 1002

    def test_heuristic_shots_from_goals(self) -> None:
        client = MagicMock()
        client.get.return_value = _match_detail_response(home_goals=2, away_goals=1)
        builder = MatchDetailsBuilder(client, match_ids=[1001])
        df = builder.build()
        row = df.iloc[0]
        # SHOTS_home = 2*5 + 7 = 17
        assert int(row["SHOTS_home"]) == 17
        # SHOTS_away = 1*5 + 7 = 12
        assert int(row["SHOTS_away"]) == 12

    def test_heuristic_possession_is_neutral(self) -> None:
        client = MagicMock()
        client.get.return_value = _match_detail_response()
        builder = MatchDetailsBuilder(client, match_ids=[1001])
        df = builder.build()
        assert float(df.iloc[0]["POSSESSION_home"]) == 50.0
        assert float(df.iloc[0]["POSSESSION_away"]) == 50.0

    def test_extract_detail_row_uses_statistics_when_present(self) -> None:
        data = {
            "id": 1001,
            "score": {"fullTime": {"home": 1, "away": 0}},
            "statistics": [
                {"type": "TOTAL_SHOTS", "home": 15, "away": 8},
                {"type": "SHOTS_ON_TARGET", "home": 6, "away": 3},
                {"type": "BALL_POSSESSION", "home": "60%", "away": "40%"},
                {"type": "PASS_ACCURACY", "home": "85%", "away": "78%"},
            ],
        }
        row = MatchDetailsBuilder._extract_detail_row(data, 1001)
        assert row["SHOTS_home"] == 15
        assert row["SHOTS_away"] == 8
        assert row["SHOTS_ON_TARGET_home"] == 6
        assert row["POSSESSION_home"] == pytest.approx(60.0)
        assert row["PASS_PCT_home"] == pytest.approx(85.0)

    def test_resumes_from_partial(self, tmp_path: Path) -> None:
        """When a partial file exists, already-fetched IDs are skipped."""
        import pandas as pd

        partial = tmp_path / "details.partial"
        existing_df = pd.DataFrame(
            [
                {
                    "MATCH_ID": 1001,
                    "GOALS_home": 1,
                    "SHOTS_home": 12,
                    "SHOTS_ON_TARGET_home": 4,
                    "POSSESSION_home": 50.0,
                    "PASS_PCT_home": 75.0,
                    "GOALS_away": 0,
                    "SHOTS_away": 7,
                    "SHOTS_ON_TARGET_away": 2,
                    "POSSESSION_away": 50.0,
                    "PASS_PCT_away": 75.0,
                }
            ]
        )
        existing_df.to_csv(partial, index=False)

        client = MagicMock()
        client.get.return_value = _match_detail_response(match_id=1002)
        builder = MatchDetailsBuilder(client, match_ids=[1001, 1002])
        df = builder.build(partial_path=partial)

        # Should contain both the partial row and the newly fetched one
        assert len(df) == 2
        # Only match 1002 should have been fetched from the API
        assert client.get.call_count == 1


class TestBuildMatchDetailsCsv:
    def test_writes_csv(self, tmp_path: Path) -> None:
        client = MagicMock()
        client.get.return_value = _match_detail_response(match_id=1001)

        out = tmp_path / "match_details.csv"
        df = build_match_details_csv(out, match_ids=[1001], client=client)

        assert out.exists()
        assert list(df.columns) == _MATCH_DETAILS_COLUMNS
        assert len(df) == 1

    def test_partial_deleted_on_success(self, tmp_path: Path) -> None:
        client = MagicMock()
        client.get.return_value = _match_detail_response(match_id=1001)

        out = tmp_path / "match_details.csv"
        build_match_details_csv(out, match_ids=[1001], client=client)

        partial = tmp_path / "match_details.csv.partial"
        assert not partial.exists()
