"""Live WC2026 tournament state adapter over football-data.org v4."""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, model_validator

from worldcup_playoff.data.client import FootballClient
from worldcup_playoff.data.crosswalk import normalize_team

logger = logging.getLogger(__name__)

_FINISHED_STATUS = "FINISHED"
_WC_TOURNAMENT = "FIFA World Cup"
_RESULTS_COLS = ("DATE", "HOME_TEAM", "AWAY_TEAM", "HOME_GOALS", "AWAY_GOALS", "TOURNAMENT", "NEUTRAL")


class LiveMatch(BaseModel):
    """A single match from the football-data.org matches endpoint.

    Fields ``date``, ``home_goals``, ``away_goals``, and ``neutral`` carry
    martj42-compatible values for use with ``live_fixtures_to_df``.
    """

    model_config = ConfigDict(extra="ignore")

    id: int
    utc_date: str = ""
    status: str
    stage: str = ""
    group: Optional[str] = None
    home_team: Optional[str] = None
    away_team: Optional[str] = None
    date: str = ""
    home_goals: Optional[int] = None
    away_goals: Optional[int] = None
    neutral: bool = True

    @model_validator(mode="before")
    @classmethod
    def _extract_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        out = dict(data)
        if "utcDate" in data:
            utc = data["utcDate"] or ""
            out["utc_date"] = utc
            out.setdefault("date", utc[:10])
        if "homeTeam" in data:
            n = (data.get("homeTeam") or {}).get("name")
            out["home_team"] = normalize_team(n) if n else None
        if "awayTeam" in data:
            n = (data.get("awayTeam") or {}).get("name")
            out["away_team"] = normalize_team(n) if n else None
        if "score" in data:
            ft = (data.get("score") or {}).get("fullTime") or {}
            out.setdefault("home_goals", ft.get("home"))
            out.setdefault("away_goals", ft.get("away"))
        return out


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


# ---------------------------------------------------------------------------
# martj42-schema conversion
# ---------------------------------------------------------------------------


def _match_to_row(m: LiveMatch) -> dict:
    return {
        "DATE": m.date,
        "HOME_TEAM": m.home_team,
        "AWAY_TEAM": m.away_team,
        "HOME_GOALS": m.home_goals,
        "AWAY_GOALS": m.away_goals,
        "TOURNAMENT": _WC_TOURNAMENT,
        "NEUTRAL": m.neutral,
    }


def _to_results_df(matches: list[LiveMatch]) -> pd.DataFrame:
    """Build a martj42 results DataFrame from a list of valid LiveMatch objects."""
    if not matches:
        return pd.DataFrame(columns=list(_RESULTS_COLS))
    rows = [_match_to_row(m) for m in matches]
    df = pd.DataFrame(rows)[list(_RESULTS_COLS)]
    df["HOME_GOALS"] = df["HOME_GOALS"].astype("Int64")
    df["AWAY_GOALS"] = df["AWAY_GOALS"].astype("Int64")
    return df


# Public alias used by the CLI layer and its test fixtures.
fetch_live_data = fetch_tournament_state


def live_fixtures_to_df(
    state: TournamentState, *, include_played: bool = False
) -> pd.DataFrame:
    """Convert TournamentState fixtures to a martj42-schema DataFrame.

    Processes ``remaining_group_fixtures`` (and optionally ``played``).
    None-team placeholder slots (unresolved knockout positions) are dropped.
    Goals are ``<NA>`` for remaining (unplayed) fixtures.
    """
    fixtures: list[LiveMatch] = list(state.remaining_group_fixtures)
    if include_played:
        fixtures += list(state.played)
    valid = [m for m in fixtures if m.home_team is not None and m.away_team is not None]
    return _to_results_df(valid)
