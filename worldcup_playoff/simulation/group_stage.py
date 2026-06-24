"""Group-stage simulation engine with FIFA tiebreak ranking.

Ingests a TournamentState, holds played results fixed, samples remaining fixtures
via an injected ScorelineSampler, and returns v4-shaped standings for all groups.
"""
from __future__ import annotations

import random as _random
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Internal types
# ─────────────────────────────────────────────────────────────────────────────

_Match = tuple[str, str, int, int]  # (home, away, home_goals, away_goals)


@dataclass
class _TeamStats:
    """Mutable per-team accumulator for group-stage statistics."""

    name: str
    pts: int = 0
    gf: int = 0
    ga: int = 0

    @property
    def gd(self) -> int:
        return self.gf - self.ga

    def add(self, scored: int, conceded: int) -> None:
        self.gf += scored
        self.ga += conceded
        if scored > conceded:
            self.pts += 3
        elif scored == conceded:
            self.pts += 1


# ─────────────────────────────────────────────────────────────────────────────
# Pure helpers (no side-effects, easily unit-testable)
# ─────────────────────────────────────────────────────────────────────────────


def _sort_key(s: _TeamStats) -> tuple[int, int, int]:
    """Descending sort key: points, goal-difference, goals-for."""
    return (-s.pts, -s.gd, -s.gf)


def _build_stats(matches: list[_Match]) -> dict[str, _TeamStats]:
    """Accumulate overall stats for all teams from a list of matches."""
    stats: dict[str, _TeamStats] = {}
    for home, away, hg, ag in matches:
        stats.setdefault(home, _TeamStats(home))
        stats.setdefault(away, _TeamStats(away))
        stats[home].add(hg, ag)
        stats[away].add(ag, hg)
    return stats


def _h2h_stats(subset: list[str], matches: list[_Match]) -> dict[str, _TeamStats]:
    """Compute mini-table stats for matches only among the tied subset."""
    tied = set(subset)
    h2h: dict[str, _TeamStats] = {t: _TeamStats(t) for t in subset}
    for home, away, hg, ag in matches:
        if home in tied and away in tied:
            h2h[home].add(hg, ag)
            h2h[away].add(ag, hg)
    return h2h


def _resolve_tied(
    tied: list[str], matches: list[_Match], rng: _random.Random
) -> list[str]:
    """H2H mini-table → coin-flip fallback for any still-tied sub-groups."""
    if len(tied) == 1:
        return tied
    h2h = _h2h_stats(tied, matches)
    by_h2h = sorted(tied, key=lambda t: _sort_key(h2h[t]))
    result: list[str] = []
    i = 0
    while i < len(by_h2h):
        j = i + 1
        while j < len(by_h2h) and _sort_key(h2h[by_h2h[j]]) == _sort_key(h2h[by_h2h[i]]):
            j += 1
        sub = by_h2h[i:j]
        if len(sub) > 1:
            rng.shuffle(sub)
        result.extend(sub)
        i = j
    return result


def _rank_group(
    teams: list[str],
    stats: dict[str, _TeamStats],
    matches: list[_Match],
    rng: _random.Random,
) -> list[str]:
    """Return teams in ranked order using the full FIFA tiebreak chain."""
    ordered = sorted(teams, key=lambda t: _sort_key(stats[t]))
    result: list[str] = []
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and _sort_key(stats[ordered[j]]) == _sort_key(stats[ordered[i]]):
            j += 1
        group = ordered[i:j]
        if len(group) > 1:
            group = _resolve_tied(group, matches, rng)
        result.extend(group)
        i = j
    return result


def _to_v4_row(name: str, stats: _TeamStats, position: int) -> dict[str, Any]:
    """Produce a football-data.org v4 standings row consumed by resolve_r32."""
    return {
        "team": {"name": name},
        "position": position,
        "points": stats.pts,
        "goalsFor": stats.gf,
        "goalsAgainst": stats.ga,
        "goalDifference": stats.gd,
    }


def _extract_scoreline(raw: Any) -> tuple[int, int]:
    """Normalise sampler output (tuple or ndarray) to (home_goals, away_goals)."""
    if isinstance(raw, tuple):
        return int(raw[0]), int(raw[1])
    if hasattr(raw, "ndim") and raw.ndim == 2:  # ndarray shape (size, 2)
        return int(raw[0, 0]), int(raw[0, 1])
    return int(raw[0]), int(raw[1])  # ndarray shape (2,)


def _get_played(state: Any) -> list[Any]:
    """Return played matches, tolerating both .played and .played_matches."""
    if hasattr(state, "played_matches"):
        return list(state.played_matches)
    return list(state.played)


def _row_to_v4(row: Any) -> dict[str, Any]:
    """Convert a Pydantic TableRow to a v4 standings row dict."""
    return {
        "team": {"name": row.team_name},
        "points": row.points,
        "goalsFor": row.goals_for,
        "goalsAgainst": row.goals_against,
        "goalDifference": row.goal_difference,
    }


def _get_standings(state: Any) -> dict[str, list[dict[str, Any]]]:
    """Normalize state.standings to {group_label: [v4_row_dict]}.

    Handles two forms:
    - dict form (test stubs): already keyed by group label with v4 row dicts.
    - list[GroupStanding] form (real TournamentState): Pydantic objects with
      .group and .table attributes; empty tables are skipped so the fallback
      to match-accumulation is preserved for the offline martj42 path.
    """
    raw = getattr(state, "standings", None)
    if not raw:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    result: dict[str, list[dict[str, Any]]] = {}
    for gs in raw:
        label, table = getattr(gs, "group", None), getattr(gs, "table", [])
        if label and table:
            result[label] = [_row_to_v4(r) for r in table]
    return result


def _stats_from_seed(rows: list[dict[str, Any]]) -> dict[str, _TeamStats]:
    """Build _TeamStats from v4 standings rows (aggregate stats, no match replay)."""
    stats: dict[str, _TeamStats] = {}
    for row in rows:
        name = row["team"]["name"]
        s = _TeamStats(name)
        s.pts = row.get("points", 0)
        s.gf = row.get("goalsFor", 0)
        s.ga = row.get("goalsAgainst", 0)
        stats[name] = s
    return stats


def _add_matches(stats: dict[str, _TeamStats], matches: list[_Match]) -> None:
    """Accumulate match results into an existing stats dict in place."""
    for home, away, hg, ag in matches:
        stats.setdefault(home, _TeamStats(home))
        stats.setdefault(away, _TeamStats(away))
        stats[home].add(hg, ag)
        stats[away].add(ag, hg)


def _stats_for_group(
    played: list[_Match],
    remaining: list[_Match],
    seed: list[dict[str, Any]] | None,
) -> dict[str, _TeamStats]:
    """Return team stats for one group, preferring seed over re-derivation.

    When *seed* is provided it is used as the authoritative aggregate base
    (e.g. official standings from the live API); only *remaining* fixtures are
    accumulated on top.  *played* is kept for the H2H tiebreak match list but
    NOT re-added to avoid double-counting against the seed.
    When *seed* is None, stats are derived from played + remaining as usual.
    """
    if seed is None:
        return _build_stats(played + remaining)
    stats = _stats_from_seed(seed)
    _add_matches(stats, remaining)
    return stats


def _append(
    groups: dict[str, list[_Match]],
    group: str,
    home: str,
    away: str,
    hg: int,
    ag: int,
) -> None:
    """Append a resolved match to the group registry."""
    groups.setdefault(group, []).append((home, away, hg, ag))


# ─────────────────────────────────────────────────────────────────────────────
# GroupStageSimulator
# ─────────────────────────────────────────────────────────────────────────────


class GroupStageSimulator:
    """Simulates remaining group matches; ranks all groups by FIFA tiebreaks."""

    def __init__(
        self,
        sampler: Any,
        rng: _random.Random,
        np_rng: np.random.Generator | None = None,
    ) -> None:
        self._sampler = sampler
        self._rng = rng
        self._np_rng = np_rng

    def simulate(self, state: Any) -> dict[str, list[dict[str, Any]]]:
        """Return v4 standings dict keyed by group label."""
        seeded = _get_standings(state)
        played_grps, remaining_grps = self._collect_split(state)
        labels = set(played_grps) | set(remaining_grps) | set(seeded)
        return {lbl: self._rank(lbl, played_grps, remaining_grps, seeded) for lbl in labels}

    def _sample_fixture(self, home: str, away: str) -> tuple[int, int]:
        """Draw one scoreline, using the injected numpy Generator when available."""
        if self._np_rng is not None:
            return cast("tuple[int, int]", self._sampler(home, away, self._np_rng))
        return _extract_scoreline(self._sampler.sample(home, away))

    def _collect_split(
        self, state: Any
    ) -> tuple[dict[str, list[_Match]], dict[str, list[_Match]]]:
        """Split state into (played_by_group, remaining_by_group)."""
        played: dict[str, list[_Match]] = {}
        remaining: dict[str, list[_Match]] = {}
        for m in _get_played(state):
            _append(played, m.group, m.home_team, m.away_team, m.home_goals, m.away_goals)
        for f in state.remaining_group_fixtures:
            hg, ag = self._sample_fixture(f.home_team, f.away_team)
            _append(remaining, f.group, f.home_team, f.away_team, hg, ag)
        return played, remaining

    def _rank(
        self,
        label: str,
        played_grps: dict[str, list[_Match]],
        remaining_grps: dict[str, list[_Match]],
        seeded: dict[str, list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        """Rank one group and emit v4 rows."""
        played = played_grps.get(label, [])
        remaining = remaining_grps.get(label, [])
        stats = _stats_for_group(played, remaining, seeded.get(label))
        ranked = _rank_group(list(stats), stats, played + remaining, self._rng)
        return [_to_v4_row(t, stats[t], pos + 1) for pos, t in enumerate(ranked)]
