"""Bracket builder: generates a World Cup knockout bracket TOML from live standings/teams."""

from __future__ import annotations

import datetime
import io
import logging
from pathlib import Path

from worldcup_playoff.config import BracketConfig, ClientConfig, Matchup
from worldcup_playoff.data.client import FootballClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Round-of-32 seed pairings (1 vs 32, 2 vs 31, ... 16 vs 17)
# These follow the standard single-elimination bracket convention where the
# strongest seed is paired with the weakest.
# ---------------------------------------------------------------------------
_ROUND_OF_32_PAIRS: list[tuple[int, int]] = [(i, 33 - i) for i in range(1, 17)]

_REQUIRED_TEAM_FIELDS = {"name", "id"}


def default_season() -> str:
    """Infer the current World Cup cycle year from today's date.

    World Cups take place every 4 years (2022, 2026, 2030…).  Return the
    nearest future (or current) World Cup year as a four-digit string.
    """
    today = datetime.date.today()
    year = today.year
    # World Cup years: 2022, 2026, 2030, ...
    base_wc_year = 2022
    offset = (year - base_wc_year) % 4
    wc_year = year if offset == 0 else year + (4 - offset)
    return str(wc_year)


def default_competition() -> str:
    """Return the football-data.org competition code for the FIFA World Cup."""
    return "WC"


class BracketBuilder:
    """Fetches the competition's qualified teams and produces a ``BracketConfig``.

    The bracket is built as a Round of 32 (16 matchups) where teams are paired
    by their position in the API response: position 1 vs 32, 2 vs 31, etc.
    If fewer than 32 teams are present the bracket is built with the available
    teams (power-of-two enforcement is validated and logged as a warning).
    """

    def __init__(
        self,
        client: FootballClient,
        season: str | None = None,
        competition: str | None = None,
    ) -> None:
        self._client = client
        self._season = season or default_season()
        self._competition = competition or default_competition()

    def build(self) -> BracketConfig:
        """Fetch teams and return a ``BracketConfig`` with Round-of-32 matchups."""
        team_names = self._fetch_team_names()
        matchups = self._pair_teams(team_names)
        bracket = BracketConfig(
            name=f"FIFA World Cup {self._season} Bracket",
            matchups=matchups,
        )
        self._validate_bracket(bracket)
        return bracket

    def _fetch_team_names(self) -> list[str]:
        """Fetch qualified/registered teams for the competition.

        Returns a list of country-name strings ordered by their API position
        (the API returns them in seed order when data is available).
        """
        try:
            data = self._client.get(
                f"/competitions/{self._competition}/teams",
                params={"season": self._season},
            )
        except Exception:
            logger.warning(
                "Failed to fetch teams for %s/%s via season param; retrying without",
                self._competition,
                self._season,
                exc_info=True,
            )
            data = self._client.get(f"/competitions/{self._competition}/teams")

        teams: list[dict[str, object]] = data.get("teams", [])
        if not teams:
            msg = (
                f"No teams returned for competition {self._competition} "
                f"season {self._season}"
            )
            raise RuntimeError(msg)

        names: list[str] = []
        for team in teams:
            name = str(team.get("name", "") or team.get("shortName", "") or "")
            if name:
                names.append(name)

        logger.info(
            "Fetched %d team names for %s/%s", len(names), self._competition, self._season
        )
        return names

    @staticmethod
    def _pair_teams(team_names: list[str]) -> list[Matchup]:
        """Create seed-based matchups from the ordered team list.

        Seed 1 plays seed N, seed 2 plays seed N-1, etc.  If the number of
        teams is not a power of two the pairing still proceeds with the
        available teams and a warning is emitted.
        """
        n = len(team_names)
        import math

        if n < 2:
            msg = f"Need at least 2 teams, got {n}"
            raise ValueError(msg)

        if n & (n - 1) != 0:
            # Round down to nearest power of two to form a valid bracket
            n_bracket = 2 ** math.floor(math.log2(n))
            logger.warning(
                "Team count %d is not a power of two; using top %d teams for bracket",
                n,
                n_bracket,
            )
            team_names = team_names[:n_bracket]
            n = n_bracket

        matchups: list[Matchup] = []
        for high_seed in range(1, n // 2 + 1):
            low_seed = n + 1 - high_seed
            home = team_names[high_seed - 1]
            away = team_names[low_seed - 1]
            matchups.append(Matchup(home=home, away=away, group=""))
        return matchups

    @staticmethod
    def _validate_bracket(bracket: BracketConfig) -> None:
        """Verify that the bracket has the expected structure.

        Raises:
            ValueError: If fewer than 2 matchups are present or team names repeat.
        """
        if len(bracket.matchups) < 2:  # noqa: PLR2004
            msg = f"Expected at least 2 matchups, got {len(bracket.matchups)}"
            raise ValueError(msg)

        all_teams = [m.home for m in bracket.matchups] + [m.away for m in bracket.matchups]
        if len(all_teams) != len(set(all_teams)):
            duplicates = {t for t in all_teams if all_teams.count(t) > 1}
            msg = f"Duplicate teams in bracket: {duplicates}"
            raise ValueError(msg)


def _serialize_bracket_toml(bracket: BracketConfig, season: str, competition: str) -> str:
    """Serialize a BracketConfig to TOML format.

    Args:
        bracket: The bracket configuration to serialize.
        season: Season label included in the header comment.
        competition: Competition code included in the header comment.

    Returns:
        TOML-formatted string.
    """
    buf = io.StringIO()
    buf.write(f"# Auto-generated from {competition} {season} standings\n")
    buf.write(f'name = "{bracket.name}"\n')

    for matchup in bracket.matchups:
        buf.write("\n[[matchups]]\n")
        buf.write(f'home = "{matchup.home}"\n')
        buf.write(f'away = "{matchup.away}"\n')
        buf.write(f'group = "{matchup.group}"\n')

    return buf.getvalue()


def generate_bracket_toml(
    output_path: Path,
    *,
    season: str | None = None,
    competition: str = "WC",
    client: FootballClient | None = None,
) -> BracketConfig:
    """Build a World Cup bracket from qualified teams and write to a TOML file.

    Args:
        output_path: Where to write the TOML file.
        season: World Cup year string (e.g. ``"2026"``). Inferred from date if ``None``.
        competition: football-data.org competition code (default ``"WC"``).
        client: Rate-limited API client. Created with defaults if ``None``.

    Returns:
        The generated ``BracketConfig``.
    """
    if client is None:
        client = FootballClient(ClientConfig())

    resolved_season = season or default_season()

    builder = BracketBuilder(client, season=resolved_season, competition=competition)
    bracket = builder.build()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = _serialize_bracket_toml(bracket, resolved_season, competition)
    output_path.write_text(content)
    logger.info("Wrote bracket to %s", output_path)
    return bracket
