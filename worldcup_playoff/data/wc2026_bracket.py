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
from typing import Any, cast

from worldcup_playoff.data.wc2026_annexc import THIRD_PLACE_COMBINATIONS

logger = logging.getLogger(__name__)

# Public surface (THIRD_PLACE_COMBINATIONS is re-exported from wc2026_annexc).
__all__ = [
    "GROUPS",
    "R32_SLOTS",
    "THIRD_PLACE_COMBINATIONS",
    "assign_thirds",
    "rank_third_places",
    "resolve_r32",
]

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

GROUPS: tuple[str, ...] = tuple("ABCDEFGHIJKL")

# Official WC2026 R32 bracket template (matches 73–88, per FIFA/Wikipedia).
# Tuple is immutable (frozen). Each entry: (home_slot, away_slot).
#
# ORDER MATTERS: the knockout simulators fold the bracket by pairing ADJACENT
# entries (winner[0] vs winner[1], winner[2] vs winner[3], …) at every round, so
# this list must be in true bracket-adjacency order — NOT FIFA match-number order.
# FIFA numbers matches by kickoff date, which is not the bracket order: e.g. the
# R16 pairs winners of M74 vs M77 and M73 vs M75, which are not adjacent by number.
# The sequence below is laid out so adjacency reproduces the official
# R16 → QF → SF → Final tree exactly (Wikipedia matches 89–104). The trailing
# comment on each line is the FIFA match number for cross-reference.
R32_SLOTS: tuple[tuple[str, str], ...] = (
    ("1E", "3ABCDF"),  # M74  Germany (1E) vs best-3rd from A/B/C/D/F
    ("1I", "3CDFGH"),  # M77  France (1I) vs best-3rd from C/D/F/G/H   →R16 M89 (W74 v W77)
    ("2A", "2B"),  # M73
    ("1F", "2C"),  # M75                                              →R16 M90 (W73 v W75)
    ("2K", "2L"),  # M83
    ("1H", "2J"),  # M84                                              →R16 M93 (W83 v W84)
    ("1D", "3BEFIJ"),  # M81  USA (1D) vs best-3rd from B/E/F/I/J
    ("1G", "3AEHIJ"),  # M82  Belgium (1G) vs best-3rd from A/E/H/I/J →R16 M94 (W81 v W82)
    ("1C", "2F"),  # M76
    ("2E", "2I"),  # M78                                              →R16 M91 (W76 v W78)
    ("1A", "3CEFHI"),  # M79  Mexico (1A) vs best-3rd from C/E/F/H/I
    ("1L", "3EHIJK"),  # M80  England (1L) vs best-3rd from E/H/I/J/K →R16 M92 (W79 v W80)
    ("1J", "2H"),  # M86  Argentina (1J) vs runner-up H
    ("2D", "2G"),  # M88                                              →R16 M95 (W86 v W88)
    ("1B", "3EFGIJ"),  # M85  Switzerland (1B) vs best-3rd from E/F/G/I/J
    ("1K", "3DEIJL"),  # M87  Colombia (1K) vs best-3rd from D/E/I/J/L →R16 M96 (W85 v W87)
)


# THIRD_PLACE_COMBINATIONS (qualifying-8-groups frozenset → {placeholder: group})
# is the official FIFA Annex C lookup, imported verbatim from wc2026_annexc.
# A generic bipartite matching only finds *a* feasible assignment, not FIFA's
# chosen one, so the published 495-row table is used to follow the real draw.


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
