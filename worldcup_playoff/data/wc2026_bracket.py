"""WC2026 Round-of-32 bracket-slotting rules.

Pure data + pure functions. No network, no FootballClient, no randomness.
Derives concrete R32 ties from final group standings.

Slot notation:
  "1X"   — group X winner
  "2X"   — group X runner-up
  "3XYZ" — best-third placeholder; letters list eligible source groups
"""

from __future__ import annotations

import logging
from itertools import combinations
from typing import Any, cast

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

GROUPS: tuple[str, ...] = tuple("ABCDEFGHIJKL")

# Each entry: (placeholder_name, frozenset_of_eligible_source_groups)
# Source: official FIFA WC2026 R32 bracket (matches 73–88).
_THIRD_SLOT_CANDIDATES: tuple[tuple[str, frozenset[str]], ...] = (
    ("3ABCDF", frozenset("ABCDF")),
    ("3CDFGH", frozenset("CDFGH")),
    ("3CEFHI", frozenset("CEFHI")),
    ("3EHIJK", frozenset("EHIJK")),
    ("3BEFIJ", frozenset("BEFIJ")),
    ("3AEHIJ", frozenset("AEHIJ")),
    ("3EFGIJ", frozenset("EFGIJ")),
    ("3DEIJL", frozenset("DEIJL")),
)

# Official WC2026 R32 bracket template (matches 73–88, per FIFA/Wikipedia).
# Tuple is immutable (frozen). Each entry: (home_slot, away_slot).
R32_SLOTS: tuple[tuple[str, str], ...] = (
    ("2A", "2B"),  # M73
    ("1E", "3ABCDF"),  # M74  Germany (1E) vs best-3rd from A/B/C/D/F
    ("1F", "2C"),  # M75
    ("1C", "2F"),  # M76
    ("1I", "3CDFGH"),  # M77  France (1I) vs best-3rd from C/D/F/G/H
    ("2E", "2I"),  # M78
    ("1A", "3CEFHI"),  # M79  Mexico (1A) vs best-3rd from C/E/F/H/I
    ("1L", "3EHIJK"),  # M80
    ("1D", "3BEFIJ"),  # M81  USA (1D) vs best-3rd from B/E/F/I/J
    ("1G", "3AEHIJ"),  # M82
    ("2K", "2L"),  # M83
    ("1H", "2J"),  # M84
    ("1B", "3EFGIJ"),  # M85
    ("1J", "2H"),  # M86  Argentina (1J) vs runner-up H
    ("1K", "3DEIJL"),  # M87
    ("2D", "2G"),  # M88
)


# ---------------------------------------------------------------------------
# Bipartite matching (private helpers)
# ---------------------------------------------------------------------------


def _augment(
    slot: int,
    adj: list[list[int]],
    match: list[int],
    seen: list[bool],
) -> bool:
    """Try to find an augmenting path for *slot*; update *match* in-place."""
    for g in adj[slot]:
        if not seen[g]:
            seen[g] = True
            if match[g] == -1 or _augment(match[g], adj, match, seen):
                match[g] = slot
                return True
    return False


def _bipartite_match(
    qualifying: frozenset[str],
    slots: tuple[tuple[str, frozenset[str]], ...],
) -> dict[str, str] | None:
    """Return slot→group assignment via augmenting paths, or None if impossible."""
    groups = sorted(qualifying)
    g_idx = {g: i for i, g in enumerate(groups)}
    adj = [[g_idx[g] for g in cands if g in qualifying] for _, cands in slots]
    match: list[int] = [-1] * len(groups)
    n = sum(_augment(s, adj, match, [False] * len(groups)) for s in range(len(slots)))
    if n < len(slots):
        return None
    slot_map = {match[i]: groups[i] for i in range(len(groups)) if match[i] != -1}
    return {slots[s][0]: slot_map[s] for s in range(len(slots))}


# ---------------------------------------------------------------------------
# THIRD_PLACE_COMBINATIONS — computed once at import (pure, deterministic)
# ---------------------------------------------------------------------------


def _compute_third_place_combinations() -> dict[frozenset[str], dict[str, str]]:
    """Enumerate all valid 8-of-12 qualifying-third combinations via bipartite matching."""
    return {
        frozenset(combo): m
        for combo in combinations(GROUPS, 8)
        if (m := _bipartite_match(frozenset(combo), _THIRD_SLOT_CANDIDATES)) is not None
    }


# Official lookup: qualifying-8-groups frozenset → {placeholder: group_letter}.
# Covers every 8-of-12 combination that admits a valid slot assignment.
THIRD_PLACE_COMBINATIONS: dict[frozenset[str], dict[str, str]] = _compute_third_place_combinations()


# ---------------------------------------------------------------------------
# Private slot-resolution helpers
# ---------------------------------------------------------------------------


def _get_row_at(table: list[dict[str, Any]], position: int) -> dict[str, Any]:
    """Return the row whose `position` field equals *position*, or fall back by index."""
    return next(
        (r for r in table if r.get("position") == position),
        table[position - 1],
    )


def _third_sort_key(group: str, row: dict[str, Any]) -> tuple[int, int, int, str]:
    """FIFA tiebreak key for third-placed teams (lower = better rank)."""
    return (
        -row.get("points", 0),
        -row.get("goalDifference", 0),
        -row.get("goalsFor", 0),
        group,  # alphabetical as the final, deterministic tiebreaker
    )


def _resolve_slot(
    slot: str,
    standings: dict[str, list[dict[str, Any]]],
    third_map: dict[str, str],
) -> str:
    """Translate a bracket slot label into a concrete team name."""
    group = third_map[slot] if slot.startswith("3") else slot[1]
    pos = 3 if slot.startswith("3") else int(slot[0])
    team_info: dict[str, Any] = _get_row_at(standings[group], pos)["team"]
    return cast(str, team_info["name"])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def rank_third_places(standings: dict[str, list[dict[str, Any]]]) -> list[str]:
    """Return the 8 qualifying third-place group letters ranked by FIFA tiebreak.

    *standings* maps group letter → list-of-table-row dicts (football-data.org v4 schema).
    """
    ranked = sorted(
        standings.items(),
        key=lambda kv: _third_sort_key(kv[0], _get_row_at(kv[1], 3)),
    )
    return [group for group, _ in ranked[:8]]


def assign_thirds(qualified_third_groups: frozenset[str]) -> dict[str, str]:
    """Map each 3X placeholder to a concrete group letter via THIRD_PLACE_COMBINATIONS."""
    return THIRD_PLACE_COMBINATIONS[qualified_third_groups]


def resolve_r32(standings: dict[str, list[dict[str, Any]]]) -> list[tuple[str, str]]:
    """Resolve 16 R32 ties from final group standings to concrete (home, away) pairs."""
    third_map = assign_thirds(frozenset(rank_third_places(standings)))
    return [
        (_resolve_slot(h, standings, third_map), _resolve_slot(a, standings, third_map))
        for h, a in R32_SLOTS
    ]
