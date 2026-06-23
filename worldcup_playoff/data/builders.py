"""Dataset builders: constructs teams.csv, matches.csv, ranking.csv, players.csv,
and match_details.csv from the football-data.org v4 REST API.

Competition codes queried for historical match data:
- WC  : FIFA World Cup (every 4 years)
- EC  : UEFA European Championship (every 4 years)
- CA  : CONMEBOL Copa América (every 4 years)
- CLI : UEFA Champions League (annual club competition, excluded by default)
- WCQ : World Cup Qualifiers (not a v4 competition code — handled per-confederation below)

For ranking/standings the builder uses competition codes that expose group-stage
standings via /competitions/{code}/standings.  Qualifiers are not structured as
league tables in v4 so the builder collects World Cup and major confederate
championship points where available.

Fallback statistics for match_details.csv
------------------------------------------
football-data.org free-tier endpoints do not expose shots, possession, or pass
accuracy per match.  When those fields are absent the builder fills them with
deterministic heuristics:

  SHOTS_home / SHOTS_away   = GOALS * 5 + 7
      (average ~12 shots/game for teams that score 1+ goals; 7 baseline for 0-goal teams)

  SHOTS_ON_TARGET_home / _away  = max(GOALS + 2, SHOTS // 3)
      (roughly a third of all shots are on target; floors at goals+2)

  POSSESSION_home / _away   = 50.0 (neutral split — no data)

  PASS_PCT_home / _away     = 75.0 (global average pass completion ~75%)

These are fixed constants, never random, so output is reproducible.
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from worldcup_playoff.config import ClientConfig
from worldcup_playoff.data.client import FootballClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# teams.csv
# ---------------------------------------------------------------------------

_TEAMS_COLUMNS = [
    "TEAM_ID",
    "NAME",
    "SHORT_NAME",
    "TLA",
    "COUNTRY",
    "COMPETITION",
]

# Competition codes whose /teams endpoint we query to harvest national-team records
_TEAM_COMPETITION_CODES = ["WC", "EC", "CA"]


class TeamsBuilder:
    """Fetches team metadata for major international competitions."""

    def __init__(
        self,
        client: FootballClient,
        competition_codes: list[str] | None = None,
    ) -> None:
        self._client = client
        self._competition_codes = competition_codes or _TEAM_COMPETITION_CODES

    def build(self) -> pd.DataFrame:
        """Fetch teams for each competition and return the deduplicated DataFrame."""
        rows: list[dict[str, Any]] = []
        seen_ids: set[int] = set()

        for code in self._competition_codes:
            fetched = self._fetch_teams(code)
            for team in fetched:
                tid = int(team.get("id", 0))
                if tid in seen_ids:
                    continue
                seen_ids.add(tid)
                rows.append(
                    {
                        "TEAM_ID": tid,
                        "NAME": team.get("name", ""),
                        "SHORT_NAME": team.get("shortName", ""),
                        "TLA": team.get("tla", ""),
                        "COUNTRY": team.get("area", {}).get("name", ""),
                        "COMPETITION": code,
                    }
                )

        if not rows:
            msg = "No teams fetched — all competition requests failed"
            raise RuntimeError(msg)

        df = pd.DataFrame(rows, columns=_TEAMS_COLUMNS)
        logger.info("Built teams DataFrame: %d unique teams", len(df))
        return df

    def _fetch_teams(self, code: str) -> list[dict[str, Any]]:
        """Fetch teams for a single competition. Returns empty list on failure."""
        try:
            data = self._client.get(f"/competitions/{code}/teams")
            teams: list[dict[str, Any]] = data.get("teams", [])
            logger.info("Fetched %d teams for competition %s", len(teams), code)
            return teams
        except Exception:
            logger.warning("Failed to fetch teams for competition %s", code, exc_info=True)
            return []


def build_teams_csv(
    output_path: Path,
    client: FootballClient | None = None,
) -> pd.DataFrame:
    """Build teams.csv and write to disk.

    Args:
        output_path: Where to write the CSV.
        client: Rate-limited API client. Created with defaults if ``None``.

    Returns:
        The assembled DataFrame.
    """
    if client is None:
        client = FootballClient(ClientConfig())

    builder = TeamsBuilder(client)
    df = builder.build()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Wrote %d teams to %s", len(df), output_path)
    return df


# ---------------------------------------------------------------------------
# matches.csv
# ---------------------------------------------------------------------------

_MATCHES_COLUMNS = [
    "MATCH_ID",
    "DATE",
    "HOME_TEAM",
    "AWAY_TEAM",
    "HOME_GOALS",
    "AWAY_GOALS",
    "COMPETITION",
    "SEASON",
]

# Competition codes iterated when building the historical matches dataset.
# These are the main international tournaments available on football-data.org v4.
_MATCH_COMPETITION_CODES = ["WC", "EC", "CA", "CL", "PL", "BL1", "SA", "PD", "FL1"]


class MatchesBuilder:
    """Fetches match results per competition per season and assembles matches.csv.

    Mirrors ``GamesBuilder`` in the NBA original.
    """

    def __init__(
        self,
        client: FootballClient,
        start_year: int = 2006,
        end_year: int | None = None,
        competition_codes: list[str] | None = None,
    ) -> None:
        self._client = client
        self._start_year = start_year
        self._end_year = end_year if end_year is not None else datetime.date.today().year
        self._competition_codes = competition_codes or _MATCH_COMPETITION_CODES

    def build(self) -> pd.DataFrame:
        """Fetch all competitions × seasons, returning the matches DataFrame."""
        frames: list[pd.DataFrame] = []

        for code in self._competition_codes:
            for year in range(self._start_year, self._end_year + 1):
                df = self._fetch_season(code, year)
                if df is not None:
                    frames.append(df)

        if not frames:
            msg = "All match fetches failed — no data to build matches.csv"
            raise RuntimeError(msg)

        result = pd.concat(frames, ignore_index=True)

        # Remove duplicate match IDs (same match may appear under multiple codes)
        result = result.drop_duplicates(subset=["MATCH_ID"]).reset_index(drop=True)

        missing = set(_MATCHES_COLUMNS) - set(result.columns)
        if missing:
            msg = f"Missing columns after assembly: {missing}"
            raise RuntimeError(msg)

        logger.info(
            "Built matches DataFrame: %d rows across %d competition-season combos",
            len(result),
            len(frames),
        )
        return result

    def _fetch_season(self, code: str, year: int) -> pd.DataFrame | None:
        """Fetch all finished matches for one competition-season. Returns None on failure."""
        try:
            data = self._client.get(
                f"/competitions/{code}/matches",
                params={"season": str(year), "status": "FINISHED"},
            )
            matches: list[dict[str, Any]] = data.get("matches", [])
            if not matches:
                return None
            rows = [self._extract_match_row(m, code, year) for m in matches]
            # Drop rows where team names could not be resolved
            valid = [r for r in rows if r is not None]
            if not valid:
                return None
            df = pd.DataFrame(valid, columns=_MATCHES_COLUMNS)
            logger.info(
                "Fetched %d matches for %s/%d", len(df), code, year
            )
            return df
        except Exception:
            logger.warning(
                "Failed to fetch matches for %s/%d", code, year, exc_info=True
            )
            return None

    @staticmethod
    def _extract_match_row(
        match: dict[str, Any],
        code: str,
        year: int,
    ) -> dict[str, Any] | None:
        """Map a single API match object to the matches.csv row schema.

        Pure transformation — no I/O.
        """
        try:
            home_team = (
                match.get("homeTeam", {}).get("name", "")
                or match.get("homeTeam", {}).get("shortName", "")
            )
            away_team = (
                match.get("awayTeam", {}).get("name", "")
                or match.get("awayTeam", {}).get("shortName", "")
            )
            if not home_team or not away_team:
                return None

            score = match.get("score", {})
            full_time = score.get("fullTime", {})
            home_goals = full_time.get("home")
            away_goals = full_time.get("away")
            if home_goals is None or away_goals is None:
                return None

            utc_date = str(match.get("utcDate", ""))
            date_str = utc_date[:10] if utc_date else ""  # "YYYY-MM-DD"

            return {
                "MATCH_ID": int(match.get("id", 0)),
                "DATE": date_str,
                "HOME_TEAM": str(home_team),
                "AWAY_TEAM": str(away_team),
                "HOME_GOALS": int(home_goals),
                "AWAY_GOALS": int(away_goals),
                "COMPETITION": code,
                "SEASON": year,
            }
        except Exception:
            logger.debug("Skipping malformed match record: %s", match, exc_info=True)
            return None


def build_matches_csv(
    output_path: Path,
    client: FootballClient | None = None,
    start_year: int = 2006,
    end_year: int | None = None,
) -> pd.DataFrame:
    """Build matches.csv and write to disk.

    Args:
        output_path: Where to write the CSV.
        client: Rate-limited API client. Created with defaults if ``None``.
        start_year: First season year to query (e.g. 2006).
        end_year: Last season year to query. Defaults to the current year.

    Returns:
        The assembled DataFrame (8 columns matching the matches.csv schema).
    """
    if client is None:
        client = FootballClient(ClientConfig())

    builder = MatchesBuilder(client, start_year=start_year, end_year=end_year)
    df = builder.build()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Wrote %d matches to %s", len(df), output_path)
    return df


# ---------------------------------------------------------------------------
# ranking.csv
# ---------------------------------------------------------------------------

_RANKING_COLUMNS = [
    "TEAM",
    "COMPETITION",
    "SEASON",
    "POSITION",
    "PLAYED",
    "WON",
    "DRAW",
    "LOST",
    "POINTS",
    "GOALS_FOR",
    "GOALS_AGAINST",
]

# Competitions with structured standings tables (group-stage tournaments)
_RANKING_COMPETITION_CODES = ["WC", "EC", "CA"]


class RankingBuilder:
    """Fetches group/league standings per season and assembles ranking.csv.

    Mirrors ``RankingBuilder`` in the NBA original.
    """

    def __init__(
        self,
        client: FootballClient,
        start_year: int = 2006,
        end_year: int | None = None,
        competition_codes: list[str] | None = None,
    ) -> None:
        self._client = client
        self._start_year = start_year
        self._end_year = end_year if end_year is not None else datetime.date.today().year
        self._competition_codes = competition_codes or _RANKING_COMPETITION_CODES

    def build(self) -> pd.DataFrame:
        """Fetch standings for all competitions × years and return the DataFrame."""
        frames: list[pd.DataFrame] = []

        for code in self._competition_codes:
            for year in range(self._start_year, self._end_year + 1):
                df = self._fetch_season(code, year)
                if df is not None:
                    frames.append(df)

        if not frames:
            msg = "All standings fetches failed — no data to build ranking.csv"
            raise RuntimeError(msg)

        result = pd.concat(frames, ignore_index=True)

        missing = set(_RANKING_COLUMNS) - set(result.columns)
        if missing:
            msg = f"Missing columns after transformation: {missing}"
            raise RuntimeError(msg)

        logger.info("Built ranking DataFrame: %d rows", len(result))
        return result

    def _fetch_season(self, code: str, year: int) -> pd.DataFrame | None:
        """Fetch standings for one competition-season. Returns None on failure."""
        try:
            data = self._client.get(
                f"/competitions/{code}/standings",
                params={"season": str(year)},
            )
            standings: list[dict[str, Any]] = data.get("standings", [])
            if not standings:
                return None
            rows: list[dict[str, Any]] = []
            for table in standings:
                table_rows: list[dict[str, Any]] = table.get("table", [])
                for entry in table_rows:
                    row = self._extract_ranking_row(entry, code, year)
                    if row is not None:
                        rows.append(row)
            if not rows:
                return None
            df = pd.DataFrame(rows, columns=_RANKING_COLUMNS)
            logger.info(
                "Fetched %d ranking rows for %s/%d", len(df), code, year
            )
            return df
        except Exception:
            logger.warning(
                "Failed to fetch standings for %s/%d", code, year, exc_info=True
            )
            return None

    @staticmethod
    def _extract_ranking_row(
        entry: dict[str, Any],
        code: str,
        year: int,
    ) -> dict[str, Any] | None:
        """Map a single standings table row to the ranking.csv schema.

        Pure transformation — no I/O.
        """
        try:
            team = entry.get("team", {})
            team_name = team.get("name", "") or team.get("shortName", "")
            if not team_name:
                return None
            return {
                "TEAM": str(team_name),
                "COMPETITION": code,
                "SEASON": year,
                "POSITION": int(entry.get("position", 0)),                "PLAYED": int(entry.get("playedGames", 0)),                "WON": int(entry.get("won", 0)),                "DRAW": int(entry.get("draw", 0)),                "LOST": int(entry.get("lost", 0)),                "POINTS": int(entry.get("points", 0)),                "GOALS_FOR": int(entry.get("goalsFor", 0)),                "GOALS_AGAINST": int(entry.get("goalsAgainst", 0)),            }
        except Exception:
            logger.debug("Skipping malformed standings entry: %s", entry, exc_info=True)
            return None


def build_ranking_csv(
    output_path: Path,
    client: FootballClient | None = None,
    start_year: int = 2006,
    end_year: int | None = None,
) -> pd.DataFrame:
    """Build ranking.csv and write to disk.

    Args:
        output_path: Where to write the CSV.
        client: Rate-limited API client. Created with defaults if ``None``.
        start_year: First season year to query.
        end_year: Last season year. Defaults to current year.

    Returns:
        The assembled DataFrame (11 columns matching the ranking.csv schema).
    """
    if client is None:
        client = FootballClient(ClientConfig())

    builder = RankingBuilder(client, start_year=start_year, end_year=end_year)
    df = builder.build()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Wrote %d ranking rows to %s", len(df), output_path)
    return df


# ---------------------------------------------------------------------------
# players.csv
# ---------------------------------------------------------------------------

_PLAYERS_COLUMNS = [
    "PLAYER_NAME",
    "PLAYER_ID",
    "NATIONALITY",
    "POSITION",
    "COMPETITION",
]


class PlayersBuilder:
    """Fetches squad rosters for a competition and assembles players.csv.

    Mirrors ``PlayersBuilder`` in the NBA original.
    """

    def __init__(self, client: FootballClient, competition: str = "WC") -> None:
        self._client = client
        self._competition = competition

    def build(self) -> pd.DataFrame:
        """Fetch teams for the competition, then squads for each team."""
        teams = self._fetch_teams()
        if not teams:
            msg = f"No teams found for competition {self._competition}"
            raise RuntimeError(msg)

        rows: list[dict[str, Any]] = []
        for team in teams:
            team_id = int(team.get("id", 0))
            squad = self._fetch_squad(team_id)
            for player in squad:
                row = self._extract_player_row(player)
                if row is not None:
                    rows.append(row)

        if not rows:
            msg = "No player data fetched"
            raise RuntimeError(msg)

        result = pd.DataFrame(rows, columns=_PLAYERS_COLUMNS)

        missing = set(_PLAYERS_COLUMNS) - set(result.columns)
        if missing:
            msg = f"Missing columns after transformation: {missing}"
            raise RuntimeError(msg)

        logger.info(
            "Built players DataFrame: %d rows for competition %s",
            len(result),
            self._competition,
        )
        return result

    def _fetch_teams(self) -> list[dict[str, Any]]:
        """Fetch team list for this competition. Failure is fatal."""
        data = self._client.get(f"/competitions/{self._competition}/teams")
        teams: list[dict[str, Any]] = data.get("teams", [])
        logger.info(
            "Fetched %d teams for competition %s", len(teams), self._competition
        )
        return teams

    def _fetch_squad(self, team_id: int) -> list[dict[str, Any]]:
        """Fetch squad for a single team. Returns empty list on failure."""
        try:
            data = self._client.get(f"/teams/{team_id}")
            squad: list[dict[str, Any]] = data.get("squad", [])
            return squad
        except Exception:
            logger.warning("Failed to fetch squad for team %d", team_id, exc_info=True)
            return []

    def _extract_player_row(
        self, player: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Map a single squad member to the players.csv row schema. Pure function."""
        try:
            name = player.get("name", "")
            if not name:
                return None
            return {
                "PLAYER_NAME": str(name),
                "PLAYER_ID": int(player.get("id", 0)),                "NATIONALITY": str(player.get("nationality", "")),
                "POSITION": str(player.get("position", "")),
                "COMPETITION": self._competition,
            }
        except Exception:
            logger.debug("Skipping malformed player record: %s", player, exc_info=True)
            return None


def build_players_csv(
    output_path: Path,
    client: FootballClient | None = None,
    competition: str = "WC",
) -> pd.DataFrame:
    """Build players.csv and write to disk.

    Args:
        output_path: Where to write the CSV.
        client: Rate-limited API client. Created with defaults if ``None``.
        competition: Competition code (e.g. ``"WC"``).

    Returns:
        The assembled DataFrame (5 columns matching the players.csv schema).
    """
    if client is None:
        client = FootballClient(ClientConfig())

    builder = PlayersBuilder(client, competition=competition)
    df = builder.build()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Wrote %d players to %s", len(df), output_path)
    return df


# ---------------------------------------------------------------------------
# match_details.csv
# ---------------------------------------------------------------------------

_MATCH_DETAILS_COLUMNS = [
    "MATCH_ID",
    "GOALS_home",
    "SHOTS_home",
    "SHOTS_ON_TARGET_home",
    "POSSESSION_home",
    "PASS_PCT_home",
    "GOALS_away",
    "SHOTS_away",
    "SHOTS_ON_TARGET_away",
    "POSSESSION_away",
    "PASS_PCT_away",
]

# ---------------------------------------------------------------------------
# Fallback heuristics (applied when the API does not return a stat)
# ---------------------------------------------------------------------------
# These are fixed, deterministic formulae — no randomness.
#
# SHOTS = goals * 5 + 7
#   Rationale: a team that scores G goals attempted roughly (G*5 + 7) shots.
#   At 0 goals → 7 shots (baseline), at 1 goal → 12 shots (FIFA avg ~12).
#
# SHOTS_ON_TARGET = max(goals + 2, shots // 3)
#   Rationale: ~1/3 of shots are on target; at minimum the goals scored + 2.
#
# POSSESSION = 50.0 (neutral default — no data available)
#
# PASS_PCT   = 75.0 (global average pass-completion rate ~75 %)
# ---------------------------------------------------------------------------


def _heuristic_shots(goals: int) -> int:
    """Estimate total shots from goals scored (deterministic fallback)."""
    return goals * 5 + 7


def _heuristic_shots_on_target(goals: int, shots: int) -> int:
    """Estimate shots on target from goals and shots (deterministic fallback)."""
    return max(goals + 2, shots // 3)


_NEUTRAL_POSSESSION: float = 50.0
_NEUTRAL_PASS_PCT: float = 75.0


class MatchDetailsBuilder:
    """Fetches per-match detail stats and assembles match_details.csv.

    Mirrors ``GamesDetailsBuilder`` in the NBA original — includes partial
    checkpointing so interrupted runs can be resumed without re-fetching.

    football-data.org free tier does not expose shots/possession/pass stats
    through the ``/matches/{id}`` endpoint.  Where values are missing the
    builder fills them with the documented deterministic heuristics above.
    """

    def __init__(
        self,
        client: FootballClient,
        match_ids: list[int],
        checkpoint_every: int = 100,
    ) -> None:
        self._client = client
        self._match_ids = match_ids
        self._checkpoint_every = checkpoint_every

    def build(self, partial_path: Path | None = None) -> pd.DataFrame:
        """Fetch detail stats for all match IDs, with optional checkpointing and resume.

        Args:
            partial_path: Path to the partial checkpoint file. If it exists the
                builder resumes from it, skipping already-fetched IDs.

        Returns:
            DataFrame with the match_details.csv schema.
        """
        existing_df: pd.DataFrame | None = None
        fetched_ids: set[int] = set()

        if partial_path is not None and partial_path.exists():
            existing_df, fetched_ids = self._load_partial(partial_path)
            logger.info("Resumed from partial: %d matches already fetched", len(fetched_ids))

        remaining = [mid for mid in self._match_ids if mid not in fetched_ids]
        logger.info("Fetching %d matches (%d skipped)", len(remaining), len(fetched_ids))

        frames: list[pd.DataFrame] = []
        if existing_df is not None and not existing_df.empty:
            frames.append(existing_df)

        failed: list[int] = []

        for i, match_id in enumerate(remaining):
            result = self._fetch_match_detail(match_id)
            if result is None:
                failed.append(match_id)
                continue
            frames.append(pd.DataFrame([result], columns=_MATCH_DETAILS_COLUMNS))

            if partial_path is not None and (i + 1) % self._checkpoint_every == 0:
                self._save_partial(frames, partial_path)
                logger.info("Checkpoint: %d/%d matches fetched", i + 1, len(remaining))

        if not frames:
            msg = "No match details fetched — all match IDs failed"
            raise RuntimeError(msg)

        result_df = pd.concat(frames, ignore_index=True)

        missing = set(_MATCH_DETAILS_COLUMNS) - set(result_df.columns)
        if missing:
            msg = f"Missing columns in match details data: {missing}"
            raise RuntimeError(msg)

        result_df = result_df[_MATCH_DETAILS_COLUMNS]

        if failed:
            logger.warning("Failed %d match IDs: %s", len(failed), failed)

        logger.info("Built match_details DataFrame: %d rows", len(result_df))
        return result_df

    def _fetch_match_detail(self, match_id: int) -> dict[str, Any] | None:
        """Fetch detail for a single match. Returns None on failure."""
        try:
            data = self._client.get(f"/matches/{match_id}")
            return self._extract_detail_row(data, match_id)
        except Exception:
            logger.warning(
                "Failed to fetch match detail for %d", match_id, exc_info=True
            )
            return None

    @staticmethod
    def _extract_detail_row(
        data: dict[str, Any], match_id: int
    ) -> dict[str, Any]:
        """Map raw API match object to the match_details.csv row schema.

        Falls back to deterministic heuristics for statistics not exposed
        by the football-data.org free tier.  Pure function — no I/O.
        """
        score = data.get("score", {})
        full_time = score.get("fullTime", {})
        home_goals = int(full_time.get("home") or 0)
        away_goals = int(full_time.get("away") or 0)

        # Attempt to read stats from the API response (paid-tier / future fields)
        odds = data.get("odds", {}) or {}

        home_shots = _heuristic_shots(home_goals)
        home_sot = _heuristic_shots_on_target(home_goals, home_shots)
        away_shots = _heuristic_shots(away_goals)
        away_sot = _heuristic_shots_on_target(away_goals, away_shots)
        home_possession: float = _NEUTRAL_POSSESSION
        away_possession: float = _NEUTRAL_POSSESSION
        home_pass_pct: float = _NEUTRAL_PASS_PCT
        away_pass_pct: float = _NEUTRAL_PASS_PCT

        # If a paid tier ever exposes these fields they override the heuristics
        stats: list[dict[str, Any]] = data.get("statistics", []) or []
        for stat in stats:
            home_val = stat.get("home")
            away_val = stat.get("away")
            stat_type = str(stat.get("type", ""))
            if stat_type == "TOTAL_SHOTS" and home_val is not None and away_val is not None:
                home_shots = int(home_val)
                away_shots = int(away_val)
            elif (
                stat_type == "SHOTS_ON_TARGET"
                and home_val is not None
                and away_val is not None
            ):
                home_sot = int(home_val)
                away_sot = int(away_val)
            elif (
                stat_type == "BALL_POSSESSION"
                and home_val is not None
                and away_val is not None
            ):
                home_possession = float(str(home_val).replace("%", ""))
                away_possession = float(str(away_val).replace("%", ""))
            elif (
                stat_type == "PASS_ACCURACY"
                and home_val is not None
                and away_val is not None
            ):
                home_pass_pct = float(str(home_val).replace("%", ""))
                away_pass_pct = float(str(away_val).replace("%", ""))

        # Suppress unused variable warning (odds kept for future extension)
        _ = odds

        return {
            "MATCH_ID": match_id,
            "GOALS_home": home_goals,
            "SHOTS_home": home_shots,
            "SHOTS_ON_TARGET_home": home_sot,
            "POSSESSION_home": home_possession,
            "PASS_PCT_home": home_pass_pct,
            "GOALS_away": away_goals,
            "SHOTS_away": away_shots,
            "SHOTS_ON_TARGET_away": away_sot,
            "POSSESSION_away": away_possession,
            "PASS_PCT_away": away_pass_pct,
        }

    @staticmethod
    def _load_partial(partial_path: Path) -> tuple[pd.DataFrame, set[int]]:
        """Load a partial CSV and extract already-fetched match IDs."""
        try:
            df = pd.read_csv(partial_path)
            fetched_ids = {int(mid) for mid in df["MATCH_ID"]}
            return df, fetched_ids
        except Exception:
            logger.warning(
                "Failed to read partial file %s, starting fresh", partial_path
            )
            return pd.DataFrame(), set()

    @staticmethod
    def _save_partial(frames: list[pd.DataFrame], partial_path: Path) -> None:
        """Concatenate accumulated frames and save to the partial checkpoint file."""
        pd.concat(frames, ignore_index=True).to_csv(partial_path, index=False)


def build_match_details_csv(
    output_path: Path,
    match_ids: list[int],
    client: FootballClient | None = None,
    checkpoint_every: int = 100,
) -> pd.DataFrame:
    """Build match_details.csv and write to disk.

    Args:
        output_path: Where to write the CSV.
        match_ids: List of integer match IDs to fetch.
        client: Rate-limited API client. Created with defaults if ``None``.
        checkpoint_every: Save partial results every N matches.

    Returns:
        The assembled DataFrame (11 columns matching the match_details.csv schema).
    """
    if client is None:
        client = FootballClient(ClientConfig())

    partial_path = output_path.parent / (output_path.name + ".partial")
    builder = MatchDetailsBuilder(client, match_ids, checkpoint_every=checkpoint_every)
    df = builder.build(partial_path=partial_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Wrote %d match detail rows to %s", len(df), output_path)

    if partial_path.exists():
        partial_path.unlink()

    return df
