"""Core per-match feature assembler for the World Cup 2026 prediction model.

Consumes a martj42 results DataFrame plus a pre-computed ``elo_df``
(columns ``home_elo``, ``away_elo``, aligned to ``sort_chronological(df)``)
and a ``TeamAbilities`` fitted by the Dixon-Coles estimator, then emits one
feature row per match including unplayed WC2026 fixtures.

**Football-only** — no socioeconomic, market-value, or odds columns ever appear.

**No forward leakage** — time-aware covariates (form, rest days, goal diff) are
computed via an incremental per-team rolling state updated AFTER each row is
assembled, so match i never sees its own or any later result.

**O(N) performance** — a single chronological pass maintains per-team state
(deque of last *window* played results + last known match date), avoiding
the O(N²) prefix scans of the raw timeaware helpers.
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

from worldcup_playoff.config import FeatureBuildConfig
from worldcup_playoff.features.confederation import (
    CONFEDERATION_MAP,
    RankingResolution,
    resolve_ranking,
)
from worldcup_playoff.features.timeaware import sort_chronological
from worldcup_playoff.simulation.poisson import TeamAbilities

if TYPE_CHECKING:
    from worldcup_playoff.data.elo import EloResult

__all__ = [
    "FEATURE_COLUMNS",
    "FeatureBuilder",
    "MatchFeatures",
    "TeamAbilities",
    "build_features",
    "to_frame",
]

logger = logging.getLogger(__name__)

# Stable, documented column order — asserted by the allow-list test.
FEATURE_COLUMNS: tuple[str, ...] = (
    "date", "home_team", "away_team", "neutral",
    "home_elo", "away_elo", "elo_diff",
    "home_attack", "home_defence", "away_attack", "away_defence",
    "home_form", "away_form",
    "home_rest_days", "away_rest_days",
    "home_goal_diff", "away_goal_diff",
    "home_confederation", "away_confederation",
    "home_ranking", "away_ranking",
    "home_goals", "away_goals",
)


# ---------------------------------------------------------------------------
# Value object — one frozen record per match
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatchFeatures:  # noqa: D101 — field names self-document
    date: str
    home_team: str
    away_team: str
    neutral: bool
    home_elo: float
    away_elo: float
    elo_diff: float
    home_attack: float
    home_defence: float
    away_attack: float
    away_defence: float
    home_form: float
    away_form: float
    home_rest_days: int | None
    away_rest_days: int | None
    home_goal_diff: float
    away_goal_diff: float
    home_confederation: str | None
    away_confederation: str | None
    home_ranking: float | None
    away_ranking: float | None
    home_goals: int | None
    away_goals: int | None


# ---------------------------------------------------------------------------
# O(N) incremental per-team rolling state
# ---------------------------------------------------------------------------


class _TeamState:
    """Rolling covariate state for one team, maintained chronologically."""

    def __init__(self, window: int, half_life: float) -> None:
        self._half = half_life
        self._played: deque[tuple[float, int, pd.Timestamp]] = deque(maxlen=window)
        self._last: pd.Timestamp | None = None

    def rest_days(self, ref: pd.Timestamp) -> int | None:
        if self._last is None or pd.isna(ref):
            return None
        return int((ref - self._last).days)

    def form(self, ref: pd.Timestamp) -> float:
        if not self._played or pd.isna(ref):
            return 0.0
        weights = [float(0.5 ** ((ref - d).days / self._half)) for _, _, d in self._played]
        total = sum(weights)
        return 0.0 if total == 0.0 else sum(p * w for p, w in zip(
            [p for p, _, _ in self._played], weights
        )) / total

    def goal_diff(self) -> float:
        return float(sum(gd for _, gd, _ in self._played))

    def after_match(self, pts: float, gd: int, date: pd.Timestamp, played: bool) -> None:
        if played:
            self._played.append((pts, gd, date))
        if pd.notna(date):
            self._last = date


# ---------------------------------------------------------------------------
# Module-level pure helpers
# ---------------------------------------------------------------------------


def _pts_gd(scored: int, conceded: int) -> tuple[float, int]:
    """Return (points, net_goal_diff) from one team's perspective."""
    gd = scored - conceded
    if scored > conceded:
        return 3.0, gd
    return (1.0, gd) if scored == conceded else (0.0, gd)


def _date_str(raw: Any) -> str:
    """Coerce any DATE cell to a YYYY-MM-DD string; '' on NaT."""
    ts = pd.to_datetime(raw, errors="coerce")
    return "" if pd.isna(ts) else ts.strftime("%Y-%m-%d")


def _na_to_none(val: Any) -> int | None:
    return None if pd.isna(val) else int(val)


def _new_state(cfg: FeatureBuildConfig) -> _TeamState:
    return _TeamState(cfg.form_window, cfg.form_half_life_days)


def _check_alignment(df: pd.DataFrame, elo_df: pd.DataFrame) -> None:
    if len(df) != len(elo_df):
        raise ValueError(
            f"elo_df length {len(elo_df)} != matches length {len(df)}: "
            "elo_df must be pre-aligned to sort_chronological(matches_df)."
        )


def _record_match(
    row: pd.Series,
    home: str,
    away: str,
    date: pd.Timestamp,
    states: dict[str, _TeamState],
) -> None:
    if pd.isna(row["HOME_GOALS"]) or pd.isna(row["AWAY_GOALS"]):
        states[home].after_match(0.0, 0, date, False)
        states[away].after_match(0.0, 0, date, False)
        return
    hg, ag = int(row["HOME_GOALS"]), int(row["AWAY_GOALS"])
    h_pts, h_gd = _pts_gd(hg, ag)
    a_pts, a_gd = _pts_gd(ag, hg)
    states[home].after_match(h_pts, h_gd, date, True)
    states[away].after_match(a_pts, a_gd, date, True)


def _update_states(
    row: pd.Series,
    states: dict[str, _TeamState],
    cfg: FeatureBuildConfig,
) -> None:
    home, away = str(row["HOME_TEAM"]), str(row["AWAY_TEAM"])
    date = pd.to_datetime(_date_str(row["DATE"]), errors="coerce")
    for team in (home, away):
        states.setdefault(team, _new_state(cfg))
    _record_match(row, home, away, date, states)


def _make_features(
    row: pd.Series,
    elo: pd.Series,
    ab: TeamAbilities,
    h_st: _TeamState,
    a_st: _TeamState,
    h_rk: RankingResolution,
    a_rk: RankingResolution,
    date: str,
) -> MatchFeatures:
    home, away = str(row["HOME_TEAM"]), str(row["AWAY_TEAM"])
    ref = pd.to_datetime(date, errors="coerce")
    h_elo, a_elo = float(elo["home_elo"]), float(elo["away_elo"])
    return MatchFeatures(
        date=date, home_team=home, away_team=away, neutral=bool(row["NEUTRAL"]),
        home_elo=h_elo, away_elo=a_elo, elo_diff=h_elo - a_elo,
        home_attack=ab.attack.get(home, 0.0), home_defence=ab.defence.get(home, 0.0),
        away_attack=ab.attack.get(away, 0.0), away_defence=ab.defence.get(away, 0.0),
        home_form=h_st.form(ref), away_form=a_st.form(ref),
        home_rest_days=h_st.rest_days(ref), away_rest_days=a_st.rest_days(ref),
        home_goal_diff=h_st.goal_diff(), away_goal_diff=a_st.goal_diff(),
        home_confederation=h_rk.confederation, away_confederation=a_rk.confederation,
        home_ranking=h_rk.value, away_ranking=a_rk.value,
        home_goals=_na_to_none(row["HOME_GOALS"]), away_goals=_na_to_none(row["AWAY_GOALS"]),
    )


def to_frame(rows: list[MatchFeatures]) -> pd.DataFrame:
    """Convert a list of MatchFeatures into a tidy DataFrame (stable column order)."""
    if not rows:
        return pd.DataFrame(columns=list(FEATURE_COLUMNS))
    df = pd.DataFrame([asdict(r) for r in rows])[list(FEATURE_COLUMNS)]
    for col in ("home_goals", "away_goals"):
        df[col] = df[col].astype("Int64")
    return df


# ---------------------------------------------------------------------------
# Public assembler
# ---------------------------------------------------------------------------


class FeatureBuilder:
    """Assemble football-only per-match covariates from pre-computed inputs.

    Parameters
    ----------
    config:     Feature-build settings (form window, half-life, staleness cutoff, seed).
    ranking:    Optional ``{team: ranking_points}`` dict; ``None`` falls back to
                the static confederation map for every team.
    confed_map: Confederation mapping (injectable for testing).
    """

    def __init__(
        self,
        config: FeatureBuildConfig,
        *,
        ranking: dict[str, float] | None = None,
        confed_map: dict[str, str] = CONFEDERATION_MAP,
    ) -> None:
        self._cfg = config
        self._ranking = ranking
        self._confed_map = confed_map

    def build(
        self,
        df: pd.DataFrame,
        elo_df: pd.DataFrame,
        abilities: TeamAbilities,
        *,
        ranking_df: dict[str, float] | None = None,
    ) -> pd.DataFrame:
        """Return one feature row per input match (includes unplayed WC2026 rows)."""
        sorted_df = sort_chronological(df)
        _check_alignment(sorted_df, elo_df)
        ranking = ranking_df if ranking_df is not None else self._ranking
        return to_frame(self._assemble(sorted_df, elo_df, abilities, ranking))

    def _assemble(
        self,
        sorted_df: pd.DataFrame,
        elo_df: pd.DataFrame,
        abilities: TeamAbilities,
        ranking: dict[str, float] | None,
    ) -> list[MatchFeatures]:
        states: dict[str, _TeamState] = {}
        rows: list[MatchFeatures] = []
        for i, (_, row) in enumerate(sorted_df.iterrows()):
            rows.append(self._row(i, row, elo_df, abilities, ranking, states))
            _update_states(row, states, self._cfg)
        return rows

    def _row(
        self,
        i: int,
        row: pd.Series,
        elo_df: pd.DataFrame,
        abilities: TeamAbilities,
        ranking: dict[str, float] | None,
        states: dict[str, _TeamState],
    ) -> MatchFeatures:
        home, away, date = str(row["HOME_TEAM"]), str(row["AWAY_TEAM"]), _date_str(row["DATE"])
        h_st = states.get(home, _new_state(self._cfg))
        a_st = states.get(away, _new_state(self._cfg))
        return _make_features(row, elo_df.iloc[i], abilities, h_st, a_st,
                               self._rank(home, date, ranking),
                               self._rank(away, date, ranking), date)

    def _rank(self, team: str, date: str, ranking: dict[str, float] | None) -> RankingResolution:
        return resolve_ranking(
            team, ranking, date, self._cfg.ranking_staleness_cutoff, confed_map=self._confed_map
        )


# ---------------------------------------------------------------------------
# High-level convenience factory
# ---------------------------------------------------------------------------


def _to_elo_df(elo_result: EloResult) -> pd.DataFrame:
    """Extract home_elo/away_elo per match from an EloResult (aligned to sorted df)."""
    return pd.DataFrame(
        [{"home_elo": d.home_elo, "away_elo": d.away_elo} for d in elo_result.match_diffs]
    )


def build_features(
    df: pd.DataFrame,
    elo: EloResult | pd.DataFrame,
    abilities: TeamAbilities,
    *,
    config: FeatureBuildConfig | None = None,
    ranking: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Assemble per-match feature covariates from pre-computed Elo and abilities.

    Parameters
    ----------
    df:         Full martj42-internal results frame (DATE/HOME_TEAM/… uppercase schema).
    elo:        Either an ``EloResult`` from ``compute_elo`` or a pre-built elo DataFrame
                with ``home_elo`` / ``away_elo`` columns aligned to ``sort_chronological(df)``.
    abilities:  Fitted Dixon-Coles abilities from ``fit_dixon_coles``.
    config:     Feature-build settings; uses ``FeatureBuildConfig()`` defaults when ``None``.
    ranking:    Optional ``{team: ranking_points}`` dict; falls back to the static
                confederation map when ``None``.

    Returns
    -------
    DataFrame with FEATURE_COLUMNS plus a ``tournament`` metadata column (one row per match,
    including unplayed WC2026 fixtures with ``home_goals``/``away_goals`` as ``<NA>``).
    """
    from worldcup_playoff.data.elo import EloResult as _EloResult  # avoid circular at import time

    cfg = config or FeatureBuildConfig()
    elo_df = _to_elo_df(elo) if isinstance(elo, _EloResult) else elo
    sorted_df = sort_chronological(df)
    features = FeatureBuilder(cfg, ranking=ranking).build(sorted_df, elo_df, abilities)
    result = features.copy()
    result["tournament"] = sorted_df["TOURNAMENT"].values
    return result
