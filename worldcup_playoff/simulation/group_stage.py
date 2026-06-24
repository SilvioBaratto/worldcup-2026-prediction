"""Group-stage simulation engine with FIFA tiebreak ranking.

Ingests a TournamentState, holds played results fixed, samples remaining fixtures
via an injected ScorelineSampler, and returns v4-shaped standings for all groups.
"""
from __future__ import annotations

import random as _random
from dataclasses import dataclass
from typing import Any

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
        groups = self._collect(state)
        return {g: self._table(ms) for g, ms in groups.items()}

    def _sample_fixture(self, home: str, away: str) -> tuple[int, int]:
        """Draw one scoreline, using the injected numpy Generator when available."""
        if self._np_rng is not None:
            return self._sampler(home, away, self._np_rng)
        return _extract_scoreline(self._sampler.sample(home, away))

    def _collect(self, state: Any) -> dict[str, list[_Match]]:
        groups: dict[str, list[_Match]] = {}
        for m in _get_played(state):
            _append(groups, m.group, m.home_team, m.away_team, m.home_goals, m.away_goals)
        for f in state.remaining_group_fixtures:
            hg, ag = self._sample_fixture(f.home_team, f.away_team)
            _append(groups, f.group, f.home_team, f.away_team, hg, ag)
        return groups

    def _table(self, matches: list[_Match]) -> list[dict[str, Any]]:
        stats = _build_stats(matches)
        ranked = _rank_group(list(stats), stats, matches, self._rng)
        return [_to_v4_row(t, stats[t], pos + 1) for pos, t in enumerate(ranked)]


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
