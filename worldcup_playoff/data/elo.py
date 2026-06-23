"""World Football Elo engine — chronological ratings from martj42 history.

Consumes the DataFrame produced by ``Martj42Loader.load_results()``.  The module
is I/O-free and network-free so tests can build DataFrames in-memory.

Design notes
------------
* ``elo_diff`` in ``MatchEloDiff`` is ``home_elo − away_elo`` without the
  ``home_advantage`` term so the emitted covariate is venue-agnostic.
* Rows are sorted by ``(date, HOME_TEAM, AWAY_TEAM, HOME_GOALS, AWAY_GOALS)``
  giving a total content-based order.  Rows that are equal on all five keys are
  truly identical matches; swapping them does not change any rating.  ``NaT``
  dates and ``NA`` goals sort last by pandas convention.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from worldcup_playoff.config import EloConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------

# Public: imported by ``worldcup_playoff.features.timeaware`` so both code paths
# share the same sort key literal and can never drift independently.
SORT_COLS = ["_d", "HOME_TEAM", "AWAY_TEAM", "HOME_GOALS", "AWAY_GOALS"]


@dataclass(frozen=True)
class EloRating:
    """Per-team, per-date Elo snapshot (post-match)."""

    team: str
    date: str
    rating: float


@dataclass(frozen=True)
class MatchEloDiff:
    """Pre-match Elo state for a single fixture.

    ``elo_diff = home_elo − away_elo`` is venue-agnostic (excludes home_advantage).
    """

    date: str
    home_team: str
    away_team: str
    home_elo: float
    away_elo: float
    elo_diff: float
    neutral: bool


@dataclass(frozen=True)
class EloResult:
    """Outcome of ``EloEngine.run``.

    Attributes
    ----------
    history:       Ordered per-team, per-date snapshots.
    match_diffs:   One entry per input row (incl. unplayed fixtures).
    final_ratings: Latest rating for every team that appeared.
    initial_rating: Default start rating — used by ``seed_wc2026`` for absent teams.
    """

    history: list[EloRating]
    match_diffs: list[MatchEloDiff]
    final_ratings: dict[str, float]
    initial_rating: float

    def history_frame(self) -> pd.DataFrame:
        """Return history as a tidy DataFrame (team / date / rating)."""
        if not self.history:
            return pd.DataFrame(columns=["team", "date", "rating"])
        return pd.DataFrame(
            [{"team": r.team, "date": r.date, "rating": r.rating} for r in self.history]
        )

    def latest_ratings_frame(self) -> pd.DataFrame:
        """Return final_ratings as a tidy DataFrame (team / rating)."""
        if not self.final_ratings:
            return pd.DataFrame(columns=["team", "rating"])
        return pd.DataFrame([{"team": t, "rating": r} for t, r in self.final_ratings.items()])


# ---------------------------------------------------------------------------
# Module-level pure math helpers
# ---------------------------------------------------------------------------


def _we(dr: float) -> float:
    """Win expectation from Elo rating difference *dr*."""
    return float(1.0 / (1.0 + 10.0 ** (-dr / 400.0)))


def _g(margin: int) -> float:
    """Goal-margin multiplier (World-Football-Elo convention)."""
    if margin <= 1:
        return 1.0
    if margin == 2:
        return 1.5
    return (11 + margin) / 8.0


def _outcome(home_goals: int, away_goals: int) -> float:
    """W value: 1.0 home win / 0.5 draw / 0.0 home loss."""
    if home_goals > away_goals:
        return 1.0
    return 0.5 if home_goals == away_goals else 0.0


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class EloEngine:
    """Compute World Football Elo ratings chronologically from a martj42 DataFrame."""

    def __init__(self, config: EloConfig) -> None:
        self._config = config

    def run(self, df: pd.DataFrame) -> EloResult:
        """Process *df* and return per-team ratings + per-match diffs."""
        ratings: dict[str, float] = {}
        history: list[EloRating] = []
        diffs: list[MatchEloDiff] = []
        for _, row in self._sort(df).iterrows():
            self._process(row, ratings, history, diffs)
        return EloResult(
            history=history,
            match_diffs=diffs,
            final_ratings={t: float(r) for t, r in ratings.items()},
            initial_rating=self._config.initial_rating,
        )

    def _sort(self, df: pd.DataFrame) -> pd.DataFrame:
        return (
            df.assign(_d=pd.to_datetime(df["DATE"], errors="coerce"))
            .sort_values(SORT_COLS, kind="stable", na_position="last")
            .drop(columns=["_d"])
            .reset_index(drop=True)
        )

    def _process(
        self,
        row: pd.Series,
        ratings: dict[str, float],
        history: list[EloRating],
        diffs: list[MatchEloDiff],
    ) -> None:
        home, away = str(row["HOME_TEAM"]), str(row["AWAY_TEAM"])
        date, neutral = str(row["DATE"]), bool(row["NEUTRAL"])
        h_elo = ratings.get(home, self._config.initial_rating)
        a_elo = ratings.get(away, self._config.initial_rating)
        diffs.append(MatchEloDiff(date, home, away, h_elo, a_elo, h_elo - a_elo, neutral))
        if pd.isna(row["HOME_GOALS"]) or pd.isna(row["AWAY_GOALS"]):
            ratings.setdefault(home, self._config.initial_rating)
            ratings.setdefault(away, self._config.initial_rating)
            history += [EloRating(home, date, ratings[home]), EloRating(away, date, ratings[away])]
            return
        new_h, new_a = self._update(
            h_elo,
            a_elo,
            int(row["HOME_GOALS"]),
            int(row["AWAY_GOALS"]),
            neutral,
            str(row["TOURNAMENT"]),
        )
        ratings[home], ratings[away] = new_h, new_a
        history += [EloRating(home, date, new_h), EloRating(away, date, new_a)]

    def _update(
        self,
        h_elo: float,
        a_elo: float,
        hg: int,
        ag: int,
        neutral: bool,
        tournament: str,
    ) -> tuple[float, float]:
        home_adv = 0.0 if neutral else self._config.home_advantage
        we = _we(h_elo - a_elo + home_adv)
        delta = self._k(tournament) * _g(abs(hg - ag)) * (_outcome(hg, ag) - we)
        return h_elo + delta, a_elo - delta

    def _k(self, tournament: str) -> int:
        """K factor: qualifier checked before world_cup for correct precedence."""
        t, c = tournament.lower(), self._config
        if any(kw.lower() in t for kw in c.qualifier_keywords):
            return c.k_qualifier
        if any(kw.lower() in t for kw in c.continental_keywords):
            return c.k_continental
        if any(kw.lower() in t for kw in c.world_cup_keywords):
            return c.k_world_cup
        return c.k_friendly


# ---------------------------------------------------------------------------
# Module-level API
# ---------------------------------------------------------------------------


def seed_wc2026(result: EloResult, teams: Iterable[str]) -> dict[str, float]:
    """Return each team's latest Elo, defaulting to *result.initial_rating* if absent.

    Pass team names explicitly so this function stays decoupled from any HTTP
    client or live-data adapter.
    """
    return {team: result.final_ratings.get(team, result.initial_rating) for team in teams}


def compute_elo(df: pd.DataFrame, config: EloConfig | None = None) -> EloResult:
    """Factory: run the Elo engine with *config* (uses defaults when ``None``)."""
    return EloEngine(config or EloConfig()).run(df)
