"""Tests for the BracketBuilder and generate_bracket_toml."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from worldcup_playoff.config import BracketConfig, Matchup, load_bracket
from worldcup_playoff.data.bracket_builder import (
    BracketBuilder,
    _serialize_bracket_toml,
    generate_bracket_toml,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_teams_response(n: int) -> dict:
    """Return a teams API response with n national teams."""
    return {
        "teams": [
            {
                "id": 100 + i,
                "name": f"Nation{i:02d}",
                "shortName": f"N{i:02d}",
            }
            for i in range(n)
        ]
    }


def _mock_client(response: dict) -> MagicMock:
    client = MagicMock()
    client.get.return_value = response
    return client


# ---------------------------------------------------------------------------
# BracketBuilder._pair_teams
# ---------------------------------------------------------------------------


class TestPairTeams:
    def test_standard_power_of_two_pairing(self) -> None:
        """4 teams produce 2 matchups: seed1 vs seed4, seed2 vs seed3."""
        teams = ["Alpha", "Beta", "Gamma", "Delta"]
        matchups = BracketBuilder._pair_teams(teams)
        assert len(matchups) == 2
        assert matchups[0].home == "Alpha"
        assert matchups[0].away == "Delta"
        assert matchups[1].home == "Beta"
        assert matchups[1].away == "Gamma"

    def test_eight_teams_produce_four_matchups(self) -> None:
        teams = [f"Team{i}" for i in range(8)]
        matchups = BracketBuilder._pair_teams(teams)
        assert len(matchups) == 4
        assert matchups[0].home == "Team0"
        assert matchups[0].away == "Team7"
        assert matchups[3].home == "Team3"
        assert matchups[3].away == "Team4"

    def test_non_power_of_two_rounds_down(self) -> None:
        """5 teams: rounded down to 4, producing 2 matchups."""
        teams = [f"Team{i}" for i in range(5)]
        matchups = BracketBuilder._pair_teams(teams)
        assert len(matchups) == 2

    def test_one_team_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2 teams"):
            BracketBuilder._pair_teams(["Solo"])

    def test_zero_teams_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2 teams"):
            BracketBuilder._pair_teams([])

    def test_all_matchups_have_empty_group(self) -> None:
        teams = [f"T{i}" for i in range(4)]
        for m in BracketBuilder._pair_teams(teams):
            assert m.group == ""


# ---------------------------------------------------------------------------
# BracketBuilder._validate_bracket
# ---------------------------------------------------------------------------


class TestValidateBracket:
    def test_valid_bracket_passes(self) -> None:
        matchups = [
            Matchup(home="Brazil", away="France"),
            Matchup(home="Germany", away="Argentina"),
        ]
        bracket = BracketConfig(name="Test", matchups=matchups)
        BracketBuilder._validate_bracket(bracket)  # should not raise

    def test_single_matchup_raises_too_few(self) -> None:
        bracket = BracketConfig(name="Test", matchups=[Matchup(home="A", away="B")])
        # Single matchup is still valid (>= 2 means at least 2 teams, 1 matchup = 2 teams)
        # Re-read the code: _validate_bracket checks len >= 2 matchups
        with pytest.raises(ValueError, match="at least 2 matchups"):
            BracketBuilder._validate_bracket(bracket)

    def test_duplicate_teams_raises(self) -> None:
        matchups = [
            Matchup(home="Brazil", away="France"),
            Matchup(home="Brazil", away="Germany"),  # Brazil duplicated
        ]
        bracket = BracketConfig(name="Test", matchups=matchups)
        with pytest.raises(ValueError, match="Duplicate teams"):
            BracketBuilder._validate_bracket(bracket)


# ---------------------------------------------------------------------------
# BracketBuilder.build — with mocked client
# ---------------------------------------------------------------------------


class TestBracketBuilderBuild:
    def test_build_with_32_teams(self) -> None:
        client = _mock_client(_make_teams_response(32))
        builder = BracketBuilder(client, season="2026", competition="WC")
        bracket = builder.build()
        assert len(bracket.matchups) == 16
        all_teams = {m.home for m in bracket.matchups} | {m.away for m in bracket.matchups}
        assert len(all_teams) == 32

    def test_build_with_8_teams(self) -> None:
        client = _mock_client(_make_teams_response(8))
        builder = BracketBuilder(client, season="2026", competition="WC")
        bracket = builder.build()
        assert len(bracket.matchups) == 4

    def test_build_raises_when_no_teams(self) -> None:
        client = _mock_client({"teams": []})
        builder = BracketBuilder(client, season="2026", competition="WC")
        with pytest.raises(RuntimeError, match="No teams returned"):
            builder.build()

    def test_build_includes_bracket_name(self) -> None:
        client = _mock_client(_make_teams_response(4))
        builder = BracketBuilder(client, season="2026", competition="WC")
        bracket = builder.build()
        assert "2026" in bracket.name

    def test_build_falls_back_when_season_param_fails(self) -> None:
        """First call with season param fails; second call without succeeds."""
        client = MagicMock()
        client.get.side_effect = [
            RuntimeError("season param not supported"),
            _make_teams_response(4),
        ]
        builder = BracketBuilder(client, season="2026", competition="WC")
        bracket = builder.build()
        assert len(bracket.matchups) == 2


# ---------------------------------------------------------------------------
# _serialize_bracket_toml
# ---------------------------------------------------------------------------


class TestSerializeBracketToml:
    def test_round_trip_via_load_bracket(self, tmp_path: Path) -> None:
        matchups = [
            Matchup(home="Brazil", away="France", group="A"),
            Matchup(home="Germany", away="Argentina", group="B"),
        ]
        bracket = BracketConfig(name="FIFA World Cup 2026 Bracket", matchups=matchups)
        toml_str = _serialize_bracket_toml(bracket, season="2026", competition="WC")

        toml_path = tmp_path / "test_bracket.toml"
        toml_path.write_text(toml_str)

        loaded = load_bracket(toml_path)
        assert len(loaded.matchups) == 2
        assert loaded.name == "FIFA World Cup 2026 Bracket"
        assert loaded.matchups[0].home == "Brazil"
        assert loaded.matchups[0].away == "France"
        assert loaded.matchups[0].group == "A"

    def test_output_contains_header_comment(self) -> None:
        matchups = [Matchup(home="A", away="B"), Matchup(home="C", away="D")]
        bracket = BracketConfig(name="Test", matchups=matchups)
        content = _serialize_bracket_toml(bracket, season="2026", competition="WC")
        assert "WC" in content
        assert "2026" in content


# ---------------------------------------------------------------------------
# generate_bracket_toml
# ---------------------------------------------------------------------------


class TestGenerateBracketToml:
    def test_writes_valid_toml_file(self, tmp_path: Path) -> None:
        client = _mock_client(_make_teams_response(8))
        output = tmp_path / "bracket.toml"
        bracket = generate_bracket_toml(output, season="2026", competition="WC", client=client)

        assert output.exists()
        assert len(bracket.matchups) == 4

        loaded = load_bracket(output)
        assert len(loaded.matchups) == 4
        all_teams = {m.home for m in loaded.matchups} | {m.away for m in loaded.matchups}
        assert len(all_teams) == 8

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        client = _mock_client(_make_teams_response(4))
        output = tmp_path / "deep" / "nested" / "bracket.toml"
        generate_bracket_toml(output, season="2026", competition="WC", client=client)
        assert output.exists()
