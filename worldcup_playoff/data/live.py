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

    @property
    def home_score(self) -> Optional[int]:
        return self.home_goals

    @property
    def away_score(self) -> Optional[int]:
        return self.away_goals

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
    remaining_group_fixtures: list[LiveMatch] = []
    standings: list[GroupStanding]

    @model_validator(mode="before")
    @classmethod
    def _accept_remaining_alias(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if "remaining" in data and "remaining_group_fixtures" not in data:
            data = {**data, "remaining_group_fixtures": data["remaining"]}
        return data

    @property
    def remaining(self) -> list[LiveMatch]:
        return self.remaining_group_fixtures


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
# Offline fallback: reconstruct WC2026 group state from the martj42 cache
# ---------------------------------------------------------------------------

_GROUP_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Knockout stage starts 2026-07-04; group stage is June 11 – July 3.
_WC2026_KNOCKOUT_START = pd.Timestamp("2026-07-04")

_LOWERCASE_TO_UPPERCASE: dict[str, str] = {
    "date": "DATE",
    "home_team": "HOME_TEAM",
    "away_team": "AWAY_TEAM",
    "home_score": "HOME_GOALS",
    "away_score": "AWAY_GOALS",
    "tournament": "TOURNAMENT",
    "neutral": "NEUTRAL",
}


def _normalize_live_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Accept raw martj42 CSV schema (lowercase) or processed schema (uppercase)."""
    rename = {k: v for k, v in _LOWERCASE_TO_UPPERCASE.items() if k in df.columns}
    return df.rename(columns=rename) if rename else df


def _reconstruct_groups(wc: pd.DataFrame) -> dict[str, str]:
    """Map each WC2026 team to a group label (A, B, …) via union-find.

    The group draw is not stored in the martj42 results, but the round-robin
    structure recovers it exactly: every group is the set of four teams that all
    play one another, i.e. a connected component of the fixture graph. Labels
    are assigned deterministically by each component's alphabetically-first team.
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for home, away in zip(wc["HOME_TEAM"], wc["AWAY_TEAM"]):
        parent[find(home)] = find(away)

    comps: dict[str, list[str]] = {}
    for team in parent:
        comps.setdefault(find(team), []).append(team)

    # Keep only 4-team components: a valid WC group has exactly four teams.
    # Oversized components indicate knockout matches merged two groups.
    valid_comps = [teams for teams in comps.values() if len(teams) == 4]

    mapping: dict[str, str] = {}
    for idx, teams in enumerate(sorted(valid_comps, key=min)):
        label = _GROUP_LABELS[idx] if idx < len(_GROUP_LABELS) else f"G{idx}"
        for team in teams:
            mapping[team] = label
    return mapping


def build_state_from_results(df: pd.DataFrame) -> TournamentState:
    """Reconstruct a WC2026 group-stage TournamentState from a martj42 results frame.

    Offline fallback for when the football-data.org live API is unreachable (e.g.
    no API key). Expects the coerced martj42 schema (``DATE``, ``HOME_TEAM``,
    ``AWAY_TEAM``, ``HOME_GOALS``, ``AWAY_GOALS``, ``TOURNAMENT``, ``NEUTRAL``).
    Filters the 2026 ``FIFA World Cup`` group matches, recovers the twelve groups
    from the round-robin structure, and splits played (both goals present) from
    remaining fixtures so ``forecast`` runs key-free.

    The recovered group labels are positional, not the official A–L draw, so the
    knockout slotting is an approximation; title odds are driven by team strength
    (Dixon-Coles abilities) rather than the exact bracket path.
    """
    df = _normalize_live_columns(df)
    empty = TournamentState(played=[], remaining_group_fixtures=[], standings=[])
    if df.empty or "TOURNAMENT" not in df.columns:
        return empty
    wc = df[df["TOURNAMENT"] == _WC_TOURNAMENT].copy()
    dates = pd.to_datetime(wc["DATE"], errors="coerce")
    wc = wc[(dates.dt.year == 2026) & (dates < _WC2026_KNOCKOUT_START)]
    if wc.empty:
        return empty

    groups = _reconstruct_groups(wc)
    played: list[LiveMatch] = []
    remaining: list[LiveMatch] = []
    for i, row in enumerate(wc.itertuples(index=False)):
        home, away = row.HOME_TEAM, row.AWAY_TEAM
        has_score = pd.notna(row.HOME_GOALS) and pd.notna(row.AWAY_GOALS)
        match = LiveMatch(
            id=i,
            status=_FINISHED_STATUS if has_score else "SCHEDULED",
            stage="GROUP_STAGE",
            group=groups.get(home),
            home_team=home,
            away_team=away,
            date=str(row.DATE)[:10],
            home_goals=int(row.HOME_GOALS) if has_score else None,
            away_goals=int(row.AWAY_GOALS) if has_score else None,
            neutral=bool(getattr(row, "NEUTRAL", True)),
        )
        (played if has_score else remaining).append(match)

    group_teams: dict[str, list[str]] = {}
    for team, label in groups.items():
        group_teams.setdefault(label, []).append(team)
    standings = [
        GroupStanding(group=f"GROUP_{label}", stage="GROUP_STAGE", table=[])
        for label in sorted(group_teams)
    ]
    return TournamentState(played=played, remaining_group_fixtures=remaining, standings=standings)


# ---------------------------------------------------------------------------
# martj42-schema conversion
# ---------------------------------------------------------------------------


def _match_to_row(m: LiveMatch) -> dict[str, Any]:
    return {
        "DATE": m.date,
        "HOME_TEAM": m.home_team,
        "AWAY_TEAM": m.away_team,
        "HOME_GOALS": m.home_goals,
        "AWAY_GOALS": m.away_goals,
        "TOURNAMENT": _WC_TOURNAMENT,
        "NEUTRAL": m.neutral,
    }


_ALIAS_COLS = ["date", "home_team", "away_team", "home_score", "away_score",
               "tournament", "neutral", "city", "country"]


def _to_results_df(matches: list[LiveMatch]) -> pd.DataFrame:
    """Build a martj42 results DataFrame from a list of valid LiveMatch objects."""
    if not matches:
        return pd.DataFrame(columns=list(_RESULTS_COLS) + _ALIAS_COLS)
    rows = [_match_to_row(m) for m in matches]
    df = pd.DataFrame(rows)[list(_RESULTS_COLS)]
    df["HOME_GOALS"] = df["HOME_GOALS"].astype("Int64")
    df["AWAY_GOALS"] = df["AWAY_GOALS"].astype("Int64")
    df["date"] = df["DATE"]
    df["home_team"] = df["HOME_TEAM"]
    df["away_team"] = df["AWAY_TEAM"]
    df["home_score"] = df["HOME_GOALS"]
    df["away_score"] = df["AWAY_GOALS"]
    df["tournament"] = df["TOURNAMENT"]
    df["neutral"] = df["NEUTRAL"]
    df["city"] = ""
    df["country"] = ""
    return df


# Public alias used by the CLI layer and its test fixtures.
fetch_live_data = fetch_tournament_state


def live_fixtures_to_df(state: TournamentState) -> pd.DataFrame:
    """Convert TournamentState fixtures to a martj42-schema DataFrame.

    Processes both ``remaining_group_fixtures`` and ``played``.
    None-team placeholder slots (unresolved knockout positions) are dropped.
    Goals are ``<NA>`` for remaining (unplayed) fixtures.
    """
    fixtures: list[LiveMatch] = list(state.remaining_group_fixtures) + list(state.played)
    valid = [m for m in fixtures if m.home_team is not None and m.away_team is not None]
    return _to_results_df(valid)
