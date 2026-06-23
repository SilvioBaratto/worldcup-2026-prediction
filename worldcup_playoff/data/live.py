"""Live WC2026 tournament state adapter over football-data.org v4."""

from __future__ import annotations

import logging
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, model_validator

from worldcup_playoff.data.client import FootballClient
from worldcup_playoff.data.crosswalk import normalize_team

logger = logging.getLogger(__name__)

_FINISHED_STATUS = "FINISHED"


class LiveMatch(BaseModel):
    """A single match from the football-data.org matches endpoint."""

    model_config = ConfigDict(extra="ignore")

    id: int
    utc_date: str = ""
    status: str
    stage: str
    group: Optional[str] = None
    home_team: Optional[str] = None
    away_team: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _extract_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        home_name = (data.get("homeTeam") or {}).get("name")
        away_name = (data.get("awayTeam") or {}).get("name")
        return {
            **data,
            "utc_date": data.get("utcDate", ""),
            "home_team": normalize_team(home_name) if home_name else None,
            "away_team": normalize_team(away_name) if away_name else None,
        }


class TableRow(BaseModel):
    """One team's row in a group standings table."""

    model_config = ConfigDict(extra="ignore")

    position: int = 0
    team_name: Optional[str] = None
    played_games: int = 0
    won: int = 0
    draw: int = 0
    lost: int = 0
    points: int = 0
    goals_for: int = 0
    goals_against: int = 0
    goal_difference: int = 0

    @model_validator(mode="before")
    @classmethod
    def _flatten(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        return {
            **data,
            "team_name": (data.get("team") or {}).get("name"),
            "played_games": data.get("playedGames", 0),
            "goals_for": data.get("goalsFor", 0),
            "goals_against": data.get("goalsAgainst", 0),
            "goal_difference": data.get("goalDifference", 0),
        }


class GroupStanding(BaseModel):
    """Standings for a single WC group."""

    model_config = ConfigDict(extra="ignore")

    group: Optional[str] = None
    stage: str = ""
    table: list[TableRow] = []


class TournamentState(BaseModel):
    """Snapshot of the WC2026 tournament as of today."""

    model_config = ConfigDict(extra="ignore")

    played: list[LiveMatch]
    remaining_group_fixtures: list[LiveMatch]
    standings: list[GroupStanding]


class LiveTournamentAdapter:
    """Fetches live WC2026 data and assembles a TournamentState."""

    def __init__(self, client: FootballClient, competition: str = "WC") -> None:
        self._client = client
        self._competition = competition

    def fetch_matches(self) -> list[LiveMatch]:
        data = self._client.get(f"/competitions/{self._competition}/matches")
        return [self._parse_match(m) for m in data.get("matches", [])]

    def fetch_standings(self) -> list[GroupStanding]:
        data = self._client.get(f"/competitions/{self._competition}/standings")
        return [GroupStanding.model_validate(s) for s in data.get("standings", [])]

    def tournament_state(self) -> TournamentState:
        matches = self.fetch_matches()
        standings = self.fetch_standings()
        played, remaining = self._split_played_remaining(matches)
        return TournamentState(
            played=played, remaining_group_fixtures=remaining, standings=standings
        )

    def _parse_match(self, raw: dict[str, Any]) -> LiveMatch:
        return LiveMatch.model_validate(raw)

    def _split_played_remaining(
        self, matches: list[LiveMatch]
    ) -> tuple[list[LiveMatch], list[LiveMatch]]:
        group_stage = [m for m in matches if m.stage == "GROUP_STAGE"]
        played = [m for m in group_stage if m.status == _FINISHED_STATUS]
        remaining = [m for m in group_stage if m.status != _FINISHED_STATUS]
        return played, remaining


def fetch_tournament_state(
    client: FootballClient | None = None,
    competition: str = "WC",
) -> TournamentState:
    """Build a TournamentState, creating a default FootballClient when none is passed."""
    if client is None:
        client = FootballClient()
    return LiveTournamentAdapter(client, competition=competition).tournament_state()
