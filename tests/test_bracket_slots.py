"""
Tests for Issue #4 — WC2026 Round-of-32 bracket-slotting rules.

Source-blind: all assertions are derived from the acceptance-criteria text only.
No implementation source was read. Every test is expected to be RED (fail) until
the implementation in worldcup_playoff/data/ is written.

Criteria covered:
  1. R32_SLOTS: frozen, 16 ties, valid slot references
  2. THIRD_PLACE_COMBINATIONS: official FIFA lookup table structure
  3. rank_third_places(): FIFA tiebreak ordering, 8 groups, deterministic
  4. assign_thirds(): placeholder resolution via lookup
  5. resolve_r32(): 16 concrete ties, 32 distinct teams, no duplicates
  6. Pure import: no FootballClient, no network/randomness at import time
  7. worldcup_playoff/data/__init__.py exports all new symbols in __all__
"""

from __future__ import annotations

import pytest
from hypothesis import assume
from hypothesis import given
from hypothesis import settings as h_settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Local test fixtures — built from the criteria, NOT from production code
# ---------------------------------------------------------------------------

_ALL_GROUPS: list[str] = list("ABCDEFGHIJKL")  # 12 WC2026 groups, canonical order


def _table_row(
    position: int,
    team_name: str,
    points: int,
    gd: int,
    gf: int,
) -> dict:
    """Build one row of a group table matching the football-data.org v4 schema."""
    return {
        "position": position,
        "team": {"name": team_name},
        "playedGames": 3,
        "points": points,
        "goalsFor": gf,
        "goalsAgainst": max(gf - gd, 0),
        "goalDifference": gd,
        "won": 0,
        "draw": 0,
        "lost": 0,
    }


def _group_table(letter: str, pts3: int, gd3: int, gf3: int) -> list[dict]:
    """4-row group table; third-placed team ('Team{letter}3') has the given stats."""
    gf3 = max(gf3, 0)
    return [
        _table_row(1, f"Team{letter}1", 9, 6, 8),
        _table_row(2, f"Team{letter}2", 6, 2, 5),
        _table_row(3, f"Team{letter}3", pts3, gd3, gf3),
        _table_row(4, f"Team{letter}4", 0, -8, 1),
    ]


def _uniform_standings(pts3: int = 3, gd3: int = 0, gf3: int = 2) -> dict[str, list[dict]]:
    """All 12 groups with identical third-place statistics (ties broken by group letter)."""
    return {g: _group_table(g, pts3, gd3, gf3) for g in _ALL_GROUPS}


def _standings_first_n_superior(n: int = 8) -> dict[str, list[dict]]:
    """
    First *n* groups (A … nth) have clearly superior third-placed teams;
    the remaining (12 - n) have clearly inferior ones.
    """
    result = {}
    for i, g in enumerate(_ALL_GROUPS):
        if i < n:
            result[g] = _group_table(g, pts3=4, gd3=2, gf3=4)
        else:
            result[g] = _group_table(g, pts3=1, gd3=-3, gf3=1)
    return result


def _standings_for_first_known_combo() -> dict[str, list[dict]]:
    """
    Build standings whose top-8 thirds exactly match the first entry in
    THIRD_PLACE_COMBINATIONS, so resolve_r32 can always resolve the bracket.
    """
    from worldcup_playoff.data import THIRD_PLACE_COMBINATIONS

    first_key = next(iter(THIRD_PLACE_COMBINATIONS))
    qualifying = frozenset(first_key)
    result = {}
    for g in _ALL_GROUPS:
        if g in qualifying:
            result[g] = _group_table(g, pts3=4, gd3=2, gf3=4)
        else:
            result[g] = _group_table(g, pts3=1, gd3=-3, gf3=1)
    return result


# ---------------------------------------------------------------------------
# Criterion 1 — R32_SLOTS: frozen data structure, 16 ties, valid slot labels
# ---------------------------------------------------------------------------


def test_when_r32_slots_imported_then_exactly_16_ties_are_present():
    from worldcup_playoff.data import R32_SLOTS

    assert len(R32_SLOTS) == 16


def test_when_r32_slots_examined_then_each_tie_has_exactly_two_slot_references():
    from worldcup_playoff.data import R32_SLOTS

    for tie in R32_SLOTS:
        assert len(tie) == 2


def test_when_r32_slots_examined_then_all_24_group_position_slots_appear_exactly_once():
    """
    Criterion: R32_SLOTS encodes all 24 group-position slots (1A–1L, 2A–2L).
    Each of the 24 must appear exactly once across the 32 slot references.
    """
    from worldcup_playoff.data import R32_SLOTS

    all_slots = [slot for tie in R32_SLOTS for slot in tie]
    expected = {f"{pos}{g}" for pos in ("1", "2") for g in "ABCDEFGHIJKL"}
    group_pos_slots = [s for s in all_slots if len(s) >= 2 and s[0] in ("1", "2")]
    assert set(group_pos_slots) == expected
    assert len(group_pos_slots) == 24  # each appears exactly once


def test_when_r32_slots_examined_then_exactly_8_third_place_placeholders_appear():
    """R32_SLOTS must include exactly 8 third-place placeholder slots."""
    from worldcup_playoff.data import R32_SLOTS

    all_slots = [slot for tie in R32_SLOTS for slot in tie]
    third_slots = [s for s in all_slots if s.startswith("3")]
    assert len(third_slots) == 8


def test_when_r32_slots_mutation_attempted_then_type_error_is_raised():
    """R32_SLOTS must be immutable (frozen tuple, frozenset, or similar)."""
    from worldcup_playoff.data import R32_SLOTS

    with pytest.raises((TypeError, AttributeError)):
        R32_SLOTS[0] = ("X", "Y")  # type: ignore[index]


def test_when_r32_slots_group_position_labels_examined_then_all_are_valid():
    """Non-third-place slots must be position digit (1 or 2) + valid group letter."""
    from worldcup_playoff.data import R32_SLOTS

    valid_pos = {"1", "2"}
    valid_groups = set("ABCDEFGHIJKL")
    for tie in R32_SLOTS:
        for slot in tie:
            if not slot.startswith("3"):
                assert slot[0] in valid_pos, f"Invalid position prefix: {slot!r}"
                assert slot[1] in valid_groups, f"Invalid group letter: {slot!r}"


# ---------------------------------------------------------------------------
# Criterion 2 — THIRD_PLACE_COMBINATIONS: official FIFA lookup table
# ---------------------------------------------------------------------------


def test_when_third_place_combinations_imported_then_it_is_non_empty():
    from worldcup_playoff.data import THIRD_PLACE_COMBINATIONS

    assert len(THIRD_PLACE_COMBINATIONS) > 0


def test_when_third_place_combinations_keys_examined_then_each_covers_exactly_8_groups():
    from worldcup_playoff.data import THIRD_PLACE_COMBINATIONS

    for combo in THIRD_PLACE_COMBINATIONS:
        key_groups = frozenset(combo)
        assert len(key_groups) == 8, f"Key must identify 8 qualifying groups; got: {combo!r}"


def test_when_third_place_combinations_keys_examined_then_all_letters_are_valid():
    from worldcup_playoff.data import THIRD_PLACE_COMBINATIONS

    valid = frozenset("ABCDEFGHIJKL")
    for combo in THIRD_PLACE_COMBINATIONS:
        for letter in frozenset(combo):
            assert letter in valid, f"Invalid group letter in combination key: {letter!r}"


def test_when_third_place_combinations_values_examined_then_each_assigns_8_placeholders():
    """Each combination value maps exactly 8 third-place placeholder slots."""
    from worldcup_playoff.data import THIRD_PLACE_COMBINATIONS

    for combo, assignment in THIRD_PLACE_COMBINATIONS.items():
        assert len(assignment) == 8, (
            f"Expected 8 slot assignments for {combo!r}, got {len(assignment)}"
        )


def test_when_third_place_combinations_values_examined_then_placeholders_start_with_3():
    from worldcup_playoff.data import THIRD_PLACE_COMBINATIONS

    for combo, assignment in THIRD_PLACE_COMBINATIONS.items():
        for placeholder in assignment:
            assert placeholder.startswith("3"), (
                f"Slot placeholder must start with '3': {placeholder!r}"
            )


def test_when_third_place_combinations_values_examined_then_group_letters_are_valid():
    from worldcup_playoff.data import THIRD_PLACE_COMBINATIONS

    valid = set("ABCDEFGHIJKL")
    for _combo, assignment in THIRD_PLACE_COMBINATIONS.items():
        for group_letter in assignment.values():
            assert group_letter in valid, f"Assigned group must be valid: {group_letter!r}"


def test_when_third_place_combinations_entry_examined_then_assigned_groups_equal_qualifying():
    """The union of assigned groups in each value must equal the qualifying set in its key."""
    from worldcup_playoff.data import THIRD_PLACE_COMBINATIONS

    for combo, assignment in THIRD_PLACE_COMBINATIONS.items():
        qualifying = frozenset(combo)
        assigned = frozenset(assignment.values())
        assert assigned == qualifying, (
            f"Assigned groups {assigned} must equal qualifying groups {qualifying}"
        )


# ---------------------------------------------------------------------------
# Criterion 3 — rank_third_places(): FIFA tiebreak, deterministic, 8 groups
# ---------------------------------------------------------------------------


def test_when_valid_standings_given_then_rank_third_places_returns_8_groups():
    from worldcup_playoff.data import rank_third_places

    result = rank_third_places(_uniform_standings())
    assert len(result) == 8


def test_when_valid_standings_given_then_returned_groups_are_distinct():
    from worldcup_playoff.data import rank_third_places

    result = rank_third_places(_uniform_standings())
    assert len(set(result)) == 8


def test_when_valid_standings_given_then_returned_groups_are_valid_letters():
    from worldcup_playoff.data import rank_third_places

    result = rank_third_places(_uniform_standings())
    valid = set("ABCDEFGHIJKL")
    for g in result:
        assert g in valid


def test_when_top_8_thirds_are_clearly_superior_then_they_qualify():
    from worldcup_playoff.data import rank_third_places

    result = rank_third_places(_standings_first_n_superior(8))
    assert set(result) == set("ABCDEFGH")


def test_when_thirds_differ_by_points_then_higher_points_ranks_first():
    """Points are the primary FIFA tiebreak criterion."""
    from worldcup_playoff.data import rank_third_places

    standings = _uniform_standings(pts3=3, gd3=0, gf3=2)
    standings["A"][2]["points"] = 7  # far ahead of every other group
    result = rank_third_places(standings)
    assert result[0] == "A"


def test_when_thirds_tie_on_points_then_better_goal_difference_ranks_higher():
    """Goal difference is the secondary tiebreak criterion."""
    from worldcup_playoff.data import rank_third_places

    standings = _uniform_standings(pts3=3, gd3=0, gf3=2)
    standings["B"][2]["goalDifference"] = 6  # far ahead on GD, same pts
    standings["B"][2]["goalsFor"] = 8
    standings["B"][2]["goalsAgainst"] = 2
    result = rank_third_places(standings)
    assert result[0] == "B"


def test_when_thirds_tie_on_points_and_gd_then_better_goals_for_ranks_higher():
    """Goals for is the tertiary tiebreak criterion."""
    from worldcup_playoff.data import rank_third_places

    standings = _uniform_standings(pts3=3, gd3=0, gf3=2)
    # Group C gets many more goals-for with the same GD (0)
    standings["C"][2]["goalsFor"] = 9
    standings["C"][2]["goalsAgainst"] = 9
    standings["C"][2]["goalDifference"] = 0
    result = rank_third_places(standings)
    assert result[0] == "C"


def test_when_same_standings_given_twice_then_rank_third_places_returns_identical_result():
    """rank_third_places must be deterministic: identical input → identical output."""
    from worldcup_playoff.data import rank_third_places

    standings = _uniform_standings()
    assert rank_third_places(standings) == rank_third_places(standings)


@given(
    st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=9),  # points for 3rd-place team
            st.integers(min_value=-9, max_value=9),  # goal difference
            st.integers(min_value=0, max_value=9),  # goals for
        ),
        min_size=12,
        max_size=12,
    )
)
@h_settings(max_examples=60)
def test_when_any_valid_standings_given_then_rank_third_places_always_returns_8_distinct_valid_groups(
    third_stats,
):
    """Invariant: rank_third_places returns exactly 8 distinct valid groups for any input."""
    from worldcup_playoff.data import rank_third_places

    standings = {
        g: _group_table(g, pts3, gd3, gf3) for g, (pts3, gd3, gf3) in zip(_ALL_GROUPS, third_stats)
    }
    result = rank_third_places(standings)
    assert len(result) == 8
    assert len(set(result)) == 8
    assert all(g in set("ABCDEFGHIJKL") for g in result)


# ---------------------------------------------------------------------------
# Criterion 4 — assign_thirds(): placeholder resolution via lookup
# ---------------------------------------------------------------------------


def _first_combo() -> frozenset[str]:
    """Return the first key from THIRD_PLACE_COMBINATIONS as a frozenset."""
    from worldcup_playoff.data import THIRD_PLACE_COMBINATIONS

    return frozenset(next(iter(THIRD_PLACE_COMBINATIONS)))


def test_when_valid_combination_given_to_assign_thirds_then_dict_with_8_entries_is_returned():
    from worldcup_playoff.data import assign_thirds

    result = assign_thirds(_first_combo())
    assert isinstance(result, dict)
    assert len(result) == 8


def test_when_valid_combination_given_to_assign_thirds_then_keys_are_third_place_placeholders():
    from worldcup_playoff.data import assign_thirds

    result = assign_thirds(_first_combo())
    for placeholder in result:
        assert placeholder.startswith("3"), f"Expected '3…' placeholder, got: {placeholder!r}"


def test_when_valid_combination_given_to_assign_thirds_then_values_are_valid_group_letters():
    from worldcup_playoff.data import assign_thirds

    result = assign_thirds(_first_combo())
    valid = set("ABCDEFGHIJKL")
    for group_letter in result.values():
        assert group_letter in valid


def test_when_valid_combination_given_to_assign_thirds_then_all_qualifying_groups_are_assigned():
    """Every group in the qualifying frozenset must appear as an assigned value."""
    from worldcup_playoff.data import assign_thirds

    combo = _first_combo()
    result = assign_thirds(combo)
    assert frozenset(result.values()) == combo


def test_when_valid_combination_given_to_assign_thirds_then_placeholders_are_in_r32_slots():
    """Each placeholder returned must reference a slot that exists in R32_SLOTS."""
    from worldcup_playoff.data import R32_SLOTS, assign_thirds

    third_slots_in_r32 = {s for tie in R32_SLOTS for s in tie if s.startswith("3")}
    result = assign_thirds(_first_combo())
    for placeholder in result:
        assert placeholder in third_slots_in_r32, (
            f"{placeholder!r} is not a known R32 third-place slot"
        )


# ---------------------------------------------------------------------------
# Criterion 5 — resolve_r32(): 16 concrete ties, 32 distinct teams, no dupes
# ---------------------------------------------------------------------------


def test_when_standings_given_to_resolve_r32_then_exactly_16_ties_are_returned():
    from worldcup_playoff.data import resolve_r32

    result = resolve_r32(_standings_for_first_known_combo())
    assert len(result) == 16


def test_when_standings_given_to_resolve_r32_then_each_tie_is_a_two_element_pair():
    from worldcup_playoff.data import resolve_r32

    result = resolve_r32(_standings_for_first_known_combo())
    for tie in result:
        assert len(tie) == 2


def test_when_standings_given_to_resolve_r32_then_exactly_32_team_slots_are_filled():
    from worldcup_playoff.data import resolve_r32

    result = resolve_r32(_standings_for_first_known_combo())
    all_teams = [team for home, away in result for team in (home, away)]
    assert len(all_teams) == 32


def test_when_standings_given_to_resolve_r32_then_all_32_teams_are_distinct():
    """No team appears more than once across the 16 ties."""
    from worldcup_playoff.data import resolve_r32

    result = resolve_r32(_standings_for_first_known_combo())
    all_teams = [team for home, away in result for team in (home, away)]
    assert len(set(all_teams)) == 32, "Duplicate team detected in resolved bracket"


def test_when_standings_given_to_resolve_r32_then_all_12_group_winners_appear():
    from worldcup_playoff.data import resolve_r32

    result = resolve_r32(_standings_for_first_known_combo())
    all_teams = {team for home, away in result for team in (home, away)}
    for g in _ALL_GROUPS:
        assert f"Team{g}1" in all_teams, f"Group winner Team{g}1 must appear in the bracket"


def test_when_standings_given_to_resolve_r32_then_all_12_runners_up_appear():
    from worldcup_playoff.data import resolve_r32

    result = resolve_r32(_standings_for_first_known_combo())
    all_teams = {team for home, away in result for team in (home, away)}
    for g in _ALL_GROUPS:
        assert f"Team{g}2" in all_teams, f"Runner-up Team{g}2 must appear in the bracket"


def test_when_standings_given_to_resolve_r32_then_exactly_8_third_placed_teams_appear():
    """The 8 qualifying thirds (not the 4 eliminated) must fill the third-place slots."""
    from worldcup_playoff.data import rank_third_places, resolve_r32

    standings = _standings_for_first_known_combo()
    result = resolve_r32(standings)
    qualifying_groups = set(rank_third_places(standings))

    all_teams = {team for home, away in result for team in (home, away)}
    # In our fixture, third-placed teams are named "Team{G}3"
    third_teams_in_bracket = {t for t in all_teams if t.endswith("3")}
    assert len(third_teams_in_bracket) == 8
    for t in third_teams_in_bracket:
        group_letter = t[4]  # "TeamA3" → index 4 → "A"
        assert group_letter in qualifying_groups, (
            f"Team {t!r} from group {group_letter!r} should not be in the bracket"
        )


@given(
    st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=9),  # points for 3rd-place team
            st.integers(min_value=-9, max_value=9),  # goal difference
            st.integers(min_value=0, max_value=9),  # goals for
        ),
        min_size=12,
        max_size=12,
    )
)
@h_settings(max_examples=30)
def test_when_standings_produce_a_known_combination_then_resolve_r32_returns_32_distinct_teams(
    third_stats,
):
    """
    Invariant: for any standings whose qualified-thirds combination exists in
    THIRD_PLACE_COMBINATIONS, resolve_r32 always returns exactly 32 distinct teams.

    Derived from criterion 5: "returns 16 concrete (home, away) pairs with 32 distinct
    teams and no duplicates" — this is a structural guarantee over all valid inputs.
    Uses hypothesis.assume to discard inputs whose combo isn't in the lookup table.
    """
    from worldcup_playoff.data import THIRD_PLACE_COMBINATIONS, rank_third_places, resolve_r32

    standings = {
        g: _group_table(g, pts3, gd3, gf3) for g, (pts3, gd3, gf3) in zip(_ALL_GROUPS, third_stats)
    }
    qualified = frozenset(rank_third_places(standings))
    assume(qualified in THIRD_PLACE_COMBINATIONS)

    result = resolve_r32(standings)
    all_teams = [team for home, away in result for team in (home, away)]
    assert len(all_teams) == 32
    assert len(set(all_teams)) == 32


# ---------------------------------------------------------------------------
# Criterion 6 — Pure import: no network, no FootballClient, no randomness
# ---------------------------------------------------------------------------


def test_when_data_package_imported_then_no_exception_is_raised():
    """The data package must be importable with no side-effects."""
    import importlib

    import worldcup_playoff.data  # noqa: F401

    importlib.reload(worldcup_playoff.data)  # re-execute module top-level — must not raise


def test_when_data_package_reloaded_then_no_http_request_is_initiated(monkeypatch):
    """No HTTP request (via requests.Session or requests.get) must fire on import."""
    import importlib

    import requests

    blocked: list[tuple] = []

    def _forbid_get(*args, **kwargs):
        blocked.append(args)
        raise AssertionError(f"HTTP GET called during import: {args}")

    def _forbid_post(*args, **kwargs):
        blocked.append(args)
        raise AssertionError(f"HTTP POST called during import: {args}")

    monkeypatch.setattr(requests, "get", _forbid_get)
    monkeypatch.setattr(requests, "post", _forbid_post)

    import worldcup_playoff.data

    importlib.reload(worldcup_playoff.data)  # must not trigger _forbid_get/_forbid_post

    assert not blocked, f"HTTP call(s) made on import: {blocked}"


def test_when_data_package_reloaded_then_football_client_is_not_constructed(monkeypatch):
    """FootballClient.__init__ must not be called as a side-effect of importing."""
    import importlib

    constructed: list[bool] = []
    try:
        from worldcup_playoff.data import client as _client_mod

        _original_init = _client_mod.FootballClient.__init__

        def _spy_init(self, *args, **kwargs):
            constructed.append(True)
            return _original_init(self, *args, **kwargs)

        monkeypatch.setattr(_client_mod.FootballClient, "__init__", _spy_init)
    except (ImportError, AttributeError):
        return  # client module absent — purity is guaranteed by absence

    import worldcup_playoff.data

    importlib.reload(worldcup_playoff.data)
    assert not constructed, "FootballClient was instantiated during package import"


# ---------------------------------------------------------------------------
# Criterion 7 — worldcup_playoff/data/__init__.py exports all new symbols
# ---------------------------------------------------------------------------


def test_when_package_all_examined_then_r32_slots_is_exported():
    import worldcup_playoff.data as pkg

    assert "R32_SLOTS" in pkg.__all__


def test_when_package_all_examined_then_third_place_combinations_is_exported():
    import worldcup_playoff.data as pkg

    assert "THIRD_PLACE_COMBINATIONS" in pkg.__all__


def test_when_package_all_examined_then_rank_third_places_is_exported():
    import worldcup_playoff.data as pkg

    assert "rank_third_places" in pkg.__all__


def test_when_package_all_examined_then_assign_thirds_is_exported():
    import worldcup_playoff.data as pkg

    assert "assign_thirds" in pkg.__all__


def test_when_package_all_examined_then_resolve_r32_is_exported():
    import worldcup_playoff.data as pkg

    assert "resolve_r32" in pkg.__all__


def test_when_new_callables_accessed_from_package_then_they_are_callable():
    import worldcup_playoff.data as pkg

    assert callable(getattr(pkg, "rank_third_places", None)), "rank_third_places must be callable"
    assert callable(getattr(pkg, "assign_thirds", None)), "assign_thirds must be callable"
    assert callable(getattr(pkg, "resolve_r32", None)), "resolve_r32 must be callable"


def test_when_data_constants_accessed_from_package_then_they_exist():
    import worldcup_playoff.data as pkg

    assert hasattr(pkg, "R32_SLOTS"), "R32_SLOTS must be importable from worldcup_playoff.data"
    assert hasattr(pkg, "THIRD_PLACE_COMBINATIONS"), (
        "THIRD_PLACE_COMBINATIONS must be importable from worldcup_playoff.data"
    )
