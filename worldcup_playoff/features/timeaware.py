"""Time-aware per-match covariate helpers — no forward leakage.

Every public function accepts a ``before_idx`` parameter and reads only rows
with ``index < before_idx``, making forward leakage structurally impossible.

The sort produced by ``sort_chronological`` is byte-identical to the Elo
engine's ordering — both import ``SORT_COLS`` from ``worldcup_playoff.data.elo``
so they can never drift independently.  This alignment guarantee means the
sorted frame is positionally 1:1 with ``EloResult.match_diffs``.

Input frame schema (from ``Martj42Loader.load_results``):
    DATE (object ISO), HOME_TEAM / AWAY_TEAM (object),
    HOME_GOALS / AWAY_GOALS (nullable Int64, <NA> for unplayed),
    TOURNAMENT (object), NEUTRAL (bool).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from worldcup_playoff.data.elo import SORT_COLS
from worldcup_playoff.simulation.poisson import decay_weight

logger = logging.getLogger(__name__)

_WIN_PTS: float = 3.0
_DRAW_PTS: float = 1.0
_LOSS_PTS: float = 0.0


# ---------------------------------------------------------------------------
# Sort — shared key with Elo engine
# ---------------------------------------------------------------------------


def sort_chronological(df: pd.DataFrame) -> pd.DataFrame:
    """Sort *df* identically to the Elo engine: stable, NaT last, reset index.

    Imports ``SORT_COLS`` from ``worldcup_playoff.data.elo`` so both code
    paths always use the same literal and any change to the key fails both.
    """
    return (
        df.assign(_d=pd.to_datetime(df["DATE"], errors="coerce"))
        .sort_values(SORT_COLS, kind="stable", na_position="last")
        .drop(columns=["_d"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _team_prior(team: str, before_idx: int, df: pd.DataFrame) -> pd.DataFrame:
    """Rows before *before_idx* where *team* appeared (home or away)."""
    prior = df[df.index < before_idx]
    return prior[(prior["HOME_TEAM"] == team) | (prior["AWAY_TEAM"] == team)]


def _played_only(rows: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows that have non-NA goals (played matches)."""
    return rows[~rows["HOME_GOALS"].isna() & ~rows["AWAY_GOALS"].isna()]


def _points_for(team: str, row: pd.Series) -> float:
    """Return 3 / 1 / 0 points from *team*'s perspective for one played match."""
    hg, ag = int(row["HOME_GOALS"]), int(row["AWAY_GOALS"])
    scored, conceded = (hg, ag) if row["HOME_TEAM"] == team else (ag, hg)
    if scored > conceded:
        return _WIN_PTS
    return _DRAW_PTS if scored == conceded else _LOSS_PTS


def _net_gd_for(team: str, row: pd.Series) -> int:
    """Net goal difference from *team*'s perspective for one played match."""
    hg, ag = int(row["HOME_GOALS"]), int(row["AWAY_GOALS"])
    return hg - ag if row["HOME_TEAM"] == team else ag - hg


def _ref_date(df: pd.DataFrame, before_idx: int) -> pd.Timestamp:
    """Parse DATE of the row at *before_idx* as a Timestamp."""
    return pd.to_datetime(df.loc[before_idx, "DATE"], errors="coerce")


def _age_days_array(ref: pd.Timestamp, played: pd.DataFrame) -> np.ndarray:
    """Days between each prior match and *ref*; NaT prior dates default to 0."""
    dates = pd.to_datetime(played["DATE"], errors="coerce")
    days = (ref - dates).dt.days.fillna(0)
    return np.asarray(days, dtype=float)


def _weighted_ppg(pts: np.ndarray, weights: np.ndarray) -> float:
    """Decay-weighted points-per-game; 0.0 when all weights are zero."""
    total = float(weights.sum())
    return 0.0 if total == 0.0 else float((weights * pts).sum() / total)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def recent_form(
    team: str,
    before_idx: int,
    sorted_df: pd.DataFrame,
    *,
    window: int = 5,
    half_life_days: float = 365.0,
) -> float:
    """Decay-weighted points-per-game over the last *window* played matches.

    Age for each prior match is anchored to ``sorted_df.loc[before_idx, 'DATE']``
    so 'time-weighted' is unambiguous and testable.  Unplayed fixtures (<NA>
    goals) are excluded.  Returns ``0.0`` (neutral) with no prior played data.
    """
    played = _played_only(_team_prior(team, before_idx, sorted_df)).tail(window)
    if played.empty:
        return 0.0
    ref = _ref_date(sorted_df, before_idx)
    ages = _age_days_array(ref, played)
    weights = np.asarray(decay_weight(ages, half_life_days))
    pts = np.array([_points_for(team, row) for _, row in played.iterrows()])
    return _weighted_ppg(pts, weights)


def rest_days(
    team: str,
    before_idx: int,
    sorted_df: pd.DataFrame,
) -> int | None:
    """Calendar days since *team*'s most recent prior dated match.

    Includes unplayed fixtures when they carry a known date (so WC2026
    scheduled matches contribute to rest-day logistics).  Returns ``None``
    on first appearance or when no prior dated match exists.
    """
    prior = _team_prior(team, before_idx, sorted_df)
    parsed = pd.to_datetime(prior["DATE"], errors="coerce")
    dated = prior[~parsed.isna()]
    if dated.empty:
        return None
    ref = _ref_date(sorted_df, before_idx)
    last = pd.to_datetime(dated["DATE"], errors="coerce").max()
    return int((ref - last).days)


def goal_difference(
    team: str,
    before_idx: int,
    sorted_df: pd.DataFrame,
    *,
    window: int = 5,
) -> float:
    """Total net goal difference from *team*'s perspective over last *window* played matches.

    Returns ``0.0`` when no prior played matches exist.
    """
    played = _played_only(_team_prior(team, before_idx, sorted_df)).tail(window)
    if played.empty:
        return 0.0
    total = sum(_net_gd_for(team, row) for _, row in played.iterrows())
    return float(total)
