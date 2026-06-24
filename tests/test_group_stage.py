"""
Source-blind example tests for GroupStageSimulator (Issues #17, #44).

Derived ONLY from the acceptance criteria for issues #17 and #44 and requirements.md.
No implementation source was read during authoring — this is the TDD Red phase.

Module target: worldcup_playoff.simulation.group_stage.GroupStageSimulator
"""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

from hypothesis import given, strategies as st

# ---------------------------------------------------------------------------
# Stub data structures — shaped from acceptance-criteria text only.
# Mirrors the TournamentState / LiveMatch shape described in the criteria and
# the football-data.org v4 API contract documented in requirements.md.
# ---------------------------------------------------------------------------


@dataclass
class LiveMatch:
    """A played group match whose scoreline is FIXED — never to be resampled."""

    group: str
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int


@dataclass
class GroupFixture:
    """A not-yet-played group match to be sampled by ScorelineSampler."""

    group: str
    home_team: str
    away_team: str


@dataclass
class TournamentState:
    played_matches: List[LiveMatch] = field(default_factory=list)
    remaining_group_fixtures: List[GroupFixture] = field(default_factory=list)
    # Issue #44 — criterion 4: pre-computed v4 standings rows seed the table.
    # Keys are group labels; values are lists of row dicts matching the
    # football-data.org v4 table[] shape (position, team.name, points, goalsFor,
    # goalsAgainst, goalDifference, playedGames, won, draw, lost).
    standings: Dict[str, List[dict]] = field(default_factory=dict)


class ScorelineSamplerSpy:
    """
    Test double for ScorelineSampler.
    Records every (home_team, away_team) pair it is asked to sample.
    Returns pre-configured scorelines cyclically.
    """

    def __init__(self, scorelines: Tuple[Tuple[int, int], ...] = ((1, 0),)):
        self._scorelines = scorelines
        self._index = 0
        self.calls: List[Tuple[str, str]] = []

    def sample(self, home_team: str, away_team: str) -> Tuple[int, int]:
        self.calls.append((home_team, away_team))
        result = self._scorelines[self._index % len(self._scorelines)]
        self._index += 1
        return result


# ---------------------------------------------------------------------------
# Import the unit under test.
# This import FAILS until worldcup_playoff/simulation/group_stage.py is written
# (intentional: Red phase of TDD — all tests below will be collected but fail).
# ---------------------------------------------------------------------------

from worldcup_playoff.simulation.group_stage import GroupStageSimulator  # noqa: E402


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _run(
    played: Union[tuple, list] = (),
    remaining: Union[tuple, list] = (),
    scorelines: Tuple[Tuple[int, int], ...] = ((1, 0),),
    seed: int = 42,
    initial_standings: Optional[dict] = None,
):
    """Run the simulator; return (standings_dict, spy)."""
    state = TournamentState(
        played_matches=list(played),
        remaining_group_fixtures=list(remaining),
        standings=initial_standings if initial_standings is not None else {},
    )
    spy = ScorelineSamplerSpy(scorelines=scorelines)
    rng = random.Random(seed)
    result = GroupStageSimulator(sampler=spy, rng=rng).simulate(state)
    return result, spy


# ===========================================================================
# Criterion 1: Played LiveMatch goals are FIXED — sampler never called for them
# ===========================================================================


class TestPlayedMatchesNeverResampled:
    def test_when_tournament_has_only_played_matches_then_sampler_is_never_called(self):
        """Played results are fixed inputs; the sampler must not be invoked for them."""
        played = [
            LiveMatch("A", "Brazil", "Germany", 2, 1),
            LiveMatch("A", "France", "Argentina", 0, 0),
            LiveMatch("A", "Brazil", "France", 1, 0),
            LiveMatch("A", "Germany", "Argentina", 1, 1),
            LiveMatch("A", "Brazil", "Argentina", 1, 0),
            LiveMatch("A", "Germany", "France", 0, 1),
        ]
        _, spy = _run(played=played, remaining=())
        assert spy.calls == [], (
            "ScorelineSampler must not be called for any already-played match; "
            f"was unexpectedly called with: {spy.calls}"
        )

    def test_when_tournament_has_remaining_fixtures_then_sampler_called_once_per_fixture(self):
        """Each remaining fixture triggers exactly one sampler call with the correct teams."""
        remaining = [
            GroupFixture("B", "Spain", "Portugal"),
            GroupFixture("B", "Morocco", "Senegal"),
        ]
        _, spy = _run(remaining=remaining, scorelines=((2, 0), (1, 1)))
        assert len(spy.calls) == 2
        assert ("Spain", "Portugal") in spy.calls
        assert ("Morocco", "Senegal") in spy.calls

    def test_when_tournament_has_both_played_and_remaining_then_sampler_called_only_for_remaining(
        self,
    ):
        """In a mixed state the sampler must touch only the unplayed fixtures."""
        played = [LiveMatch("C", "USA", "Mexico", 1, 0)]
        remaining = [GroupFixture("C", "Canada", "Honduras")]
        _, spy = _run(played=played, remaining=remaining, scorelines=((0, 1),))
        assert len(spy.calls) == 1
        assert spy.calls[0] == ("Canada", "Honduras")


# ===========================================================================
# Criterion 3: FIFA tiebreak chain — points → GD → GF → H2H → coin-flip
# ===========================================================================


class TestFifaTiebreakChain:
    """Each test isolates exactly one rung of the tiebreak chain."""

    # --- Rung 1: Points ---

    def test_when_teams_differ_in_points_then_points_determines_ranking(self):
        """
        4-team round-robin: Alpha 9pts > Beta 6pts > Gamma 3pts > Delta 0pts.
        Points alone must produce this ordering.
        """
        played = [
            LiveMatch("G1", "Alpha", "Beta", 2, 0),
            LiveMatch("G1", "Alpha", "Gamma", 2, 0),
            LiveMatch("G1", "Alpha", "Delta", 2, 0),
            LiveMatch("G1", "Beta", "Gamma", 1, 0),
            LiveMatch("G1", "Beta", "Delta", 1, 0),
            LiveMatch("G1", "Gamma", "Delta", 1, 0),
        ]
        standings, _ = _run(played=played)
        pos = {e["team"]["name"]: e["position"] for e in standings["G1"]}
        assert pos["Alpha"] < pos["Beta"] < pos["Gamma"] < pos["Delta"], (
            f"Expected Alpha<Beta<Gamma<Delta by points; got positions {pos}"
        )

    # --- Rung 2: Goal difference ---

    def test_when_teams_equal_on_points_then_goal_difference_breaks_tie(self):
        """
        3-team group: A1 and A2 both have 4 pts but A1 has better goal difference.
        A1 should rank above A2.

        Fixtures (group G2):
          A1 vs A2: draw 1-1    → A1: 1pt,  GF=1, GA=1
          A1 vs A3: A1 wins 3-0 → A1: +3pt, GF=3, GA=0
          A2 vs A3: A2 wins 1-0 → A2: +3pt, GF=1, GA=0

        A1 total: 4pts, GF=4, GA=1, GD=+3
        A2 total: 4pts, GF=2, GA=1, GD=+1
        A3 total: 0pts
        """
        played = [
            LiveMatch("G2", "A1", "A2", 1, 1),
            LiveMatch("G2", "A1", "A3", 3, 0),
            LiveMatch("G2", "A2", "A3", 1, 0),
        ]
        standings, _ = _run(played=played)
        pos = {e["team"]["name"]: e["position"] for e in standings["G2"]}
        assert pos["A1"] < pos["A2"], (
            f"A1 (GD=+3) must rank above A2 (GD=+1) when both have 4 pts; got {pos}"
        )

    # --- Rung 3: Goals for ---

    def test_when_teams_equal_on_points_and_gd_then_goals_for_breaks_tie(self):
        """
        3-team group: X and Y both have 4 pts and GD=+2, but X has GF=4 vs Y's GF=3.
        X should rank above Y.

        Fixtures (group G3):
          X vs Y: draw 1-1       → X: 1pt,  GF=1, GA=1; Y: 1pt,  GF=1, GA=1
          X vs Z: X wins 3-1     → X: +3pt, GF=3, GA=1; Z: 0pts, GF=1, GA=3
          Y vs Z: Y wins 2-0     → Y: +3pt, GF=2, GA=0; Z: 0pts, GF=0, GA=2

        X total: 4pts, GF=4, GA=2, GD=+2
        Y total: 4pts, GF=3, GA=1, GD=+2   ← same GD, different GF
        """
        played = [
            LiveMatch("G3", "X", "Y", 1, 1),
            LiveMatch("G3", "X", "Z", 3, 1),
            LiveMatch("G3", "Y", "Z", 2, 0),
        ]
        standings, _ = _run(played=played)
        pos = {e["team"]["name"]: e["position"] for e in standings["G3"]}
        assert pos["X"] < pos["Y"], (
            f"X (GF=4) must rank above Y (GF=3) when both 4pts GD=+2; got {pos}"
        )

    # --- Rung 4: Head-to-head ---

    def test_when_teams_equal_on_points_gd_gf_then_h2h_breaks_tie(self):
        """
        4-team group where P and Q are identical on points / GD / GF overall
        but P beat Q head-to-head — P must rank above Q.

        Fixtures (group G4):
          P vs Q: P wins 1-0   → P: 3pts, GF=1, GA=0; Q: 0pts, GF=0, GA=1
          P vs R: draw 1-1     → P: 1pt,  GF=1, GA=1; R: 1pt,  GF=1, GA=1
          P vs S: P loses 0-1  → P: 0pts, GF=0, GA=1; S: 3pts, GF=1, GA=0
          Q vs R: Q wins 2-1   → Q: 3pts, GF=2, GA=1; R: 0pts, GF=1, GA=2
          Q vs S: draw 0-0     → Q: 1pt,  GF=0, GA=0; S: 1pt,  GF=0, GA=0
          R vs S: R wins 2-0   → R: 3pts, GF=2, GA=0; S: 0pts, GF=0, GA=2

        P total: 4pts, GF=2, GA=2, GD=0
        Q total: 4pts, GF=2, GA=2, GD=0   ← identical to P overall
        R total: 4pts, GF=4, GA=3, GD=+1  → ranks 1st by GD
        S total: 4pts, GF=1, GA=2, GD=-1  → ranks last by GD
        H2H P vs Q: P won 1-0 → P ranks above Q.
        """
        played = [
            LiveMatch("G4", "P", "Q", 1, 0),
            LiveMatch("G4", "P", "R", 1, 1),
            LiveMatch("G4", "P", "S", 0, 1),
            LiveMatch("G4", "Q", "R", 2, 1),
            LiveMatch("G4", "Q", "S", 0, 0),
            LiveMatch("G4", "R", "S", 2, 0),
        ]
        standings, _ = _run(played=played)
        pos = {e["team"]["name"]: e["position"] for e in standings["G4"]}
        assert pos["P"] < pos["Q"], (
            f"P (beat Q h2h 1-0) must rank above Q when both 4pts GD=0 GF=2; got {pos}"
        )

    # --- Rung 5: Coin-flip with injected RNG ---

    def test_when_all_tiebreakers_exhausted_then_injected_rng_produces_varying_outcomes(self):
        """
        A circular 3-team group (P→Q→Z→P, all 1-0) produces an unbreakable
        3-way tie on all statistical rungs. Coin-flip must use the injected RNG:
        across 30 different seeds at least two distinct top-team orderings must appear.
        """
        played = [
            LiveMatch("G5", "P", "Q", 1, 0),
            LiveMatch("G5", "Q", "Z", 1, 0),
            LiveMatch("G5", "Z", "P", 1, 0),
        ]
        state = TournamentState(played_matches=played, remaining_group_fixtures=[])
        orderings = set()
        for seed in range(30):
            spy = ScorelineSamplerSpy()
            rng = random.Random(seed)
            result = GroupStageSimulator(sampler=spy, rng=rng).simulate(state)
            top = tuple(
                e["team"]["name"] for e in sorted(result["G5"], key=lambda e: e["position"])[:2]
            )
            orderings.add(top)

        assert len(orderings) > 1, (
            "With 30 seeds on an unbreakable 3-way tie, at least 2 distinct orderings "
            "must occur — if all seeds agree, the injected RNG is not being used"
        )

    def test_when_same_injected_seed_but_different_global_random_state_then_result_is_identical(
        self,
    ):
        """
        The simulator must use ONLY the injected RNG, not the global random module.
        With the same injected seed but different global random seeds, results must agree.
        """
        played = [
            LiveMatch("G6", "T1", "T2", 1, 0),
            LiveMatch("G6", "T2", "T3", 1, 0),
            LiveMatch("G6", "T3", "T1", 1, 0),
        ]
        state = TournamentState(played_matches=played, remaining_group_fixtures=[])

        def run_with_global_seed(global_seed: int, injected_seed: int):
            random.seed(global_seed)
            spy = ScorelineSamplerSpy()
            rng = random.Random(injected_seed)
            return GroupStageSimulator(sampler=spy, rng=rng).simulate(state)

        result_a = run_with_global_seed(global_seed=1, injected_seed=42)
        result_b = run_with_global_seed(global_seed=9999, injected_seed=42)

        pos_a = {e["team"]["name"]: e["position"] for e in result_a["G6"]}
        pos_b = {e["team"]["name"]: e["position"] for e in result_b["G6"]}
        assert pos_a == pos_b, (
            "Changing the global random seed must not affect results when the injected "
            "RNG seed is held constant — the simulator must use only the injected RNG"
        )


# ===========================================================================
# Criterion 4: Output shape matches v4 dict schema
# ===========================================================================


class TestOutputShape:
    def test_when_simulation_runs_then_result_is_a_dict_keyed_by_group_label(self):
        """simulate() must return a dict; played group labels must be present as keys."""
        played = [LiveMatch("H1", "Spain", "Morocco", 2, 0)]
        standings, _ = _run(played=played)
        assert isinstance(standings, dict), "simulate() must return a dict"
        assert "H1" in standings

    def test_when_group_has_four_teams_then_standings_list_contains_four_entries(self):
        """A 4-team group must produce exactly 4 entries in the standings list."""
        played = [
            LiveMatch("H2", "Brazil", "Germany", 1, 0),
            LiveMatch("H2", "France", "Argentina", 2, 2),
            LiveMatch("H2", "Brazil", "France", 0, 0),
            LiveMatch("H2", "Germany", "Argentina", 1, 1),
            LiveMatch("H2", "Brazil", "Argentina", 1, 0),
            LiveMatch("H2", "Germany", "France", 0, 1),
        ]
        standings, _ = _run(played=played)
        group = standings["H2"]
        assert isinstance(group, list)
        assert len(group) == 4, f"Expected 4 entries for a 4-team group; got {len(group)}"

    def test_when_simulation_runs_then_each_entry_has_required_v4_fields(self):
        """
        Each entry must contain at minimum: team.name, position, points,
        goalDifference, goalsFor — matching the football-data.org v4 standings shape
        (table[] row: position, team{id,name}, points, goalsFor, goalDifference, ...).
        """
        played = [
            LiveMatch("H3", "Netherlands", "Ecuador", 2, 1),
            LiveMatch("H3", "Senegal", "Qatar", 1, 0),
            LiveMatch("H3", "Netherlands", "Senegal", 2, 0),
            LiveMatch("H3", "Ecuador", "Qatar", 2, 0),
            LiveMatch("H3", "Netherlands", "Qatar", 2, 0),
            LiveMatch("H3", "Ecuador", "Senegal", 1, 2),
        ]
        standings, _ = _run(played=played)
        for entry in standings["H3"]:
            assert "team" in entry, f"Missing 'team' key in {entry}"
            assert "name" in entry["team"], f"Missing 'team.name' in {entry}"
            assert "position" in entry, f"Missing 'position' in {entry}"
            assert "points" in entry, f"Missing 'points' in {entry}"
            assert "goalDifference" in entry, f"Missing 'goalDifference' in {entry}"
            assert "goalsFor" in entry, f"Missing 'goalsFor' in {entry}"

    def test_when_simulation_runs_then_positions_are_one_indexed_unique_and_contiguous(self):
        """Positions within a group must be exactly {1, 2, 3, ..., N} — no gaps, no dupes."""
        played = [
            LiveMatch("H4", "X", "Y", 2, 0),
            LiveMatch("H4", "X", "Z", 1, 0),
            LiveMatch("H4", "Y", "Z", 1, 0),
        ]
        standings, _ = _run(played=played)
        positions = sorted(e["position"] for e in standings["H4"])
        assert positions == [1, 2, 3], (
            f"Positions must be [1, 2, 3] for a 3-team group; got {positions}"
        )

    def test_when_multiple_groups_exist_in_tournament_state_then_all_appear_in_output(self):
        """Every group referenced in played_matches must be a key in the output dict."""
        played = [
            LiveMatch("H5", "USA", "Mexico", 1, 0),
            LiveMatch("H6", "Spain", "Portugal", 2, 1),
        ]
        standings, _ = _run(played=played)
        assert "H5" in standings
        assert "H6" in standings


# ===========================================================================
# Criterion 3 (Issue #44): simulate() covers all 12 WC2026 groups
# ===========================================================================


class TestTwelveGroupCoverage:
    """WC2026 has 12 groups (A–L). simulate() must return an entry for each."""

    def test_when_state_contains_twelve_groups_then_all_twelve_keys_present_in_output(self):
        """
        Issue #44 criterion 3: "returns a v4 standings dict for every group (12 groups)".
        Provide one match per WC2026-style group label (GROUP_A … GROUP_L) and assert
        that all 12 are present as top-level keys in the simulate() return value.
        """
        group_labels = [f"GROUP_{c}" for c in "ABCDEFGHIJKL"]
        played = [
            LiveMatch(label, f"Home_{label}", f"Away_{label}", 1, 0) for label in group_labels
        ]
        result, _ = _run(played=played)
        missing = [lbl for lbl in group_labels if lbl not in result]
        assert missing == [], f"simulate() must return a key for every group; missing: {missing}"
        assert len(result) == 12, (
            f"Expected exactly 12 group keys; got {len(result)}: {sorted(result.keys())}"
        )

    def test_when_twelve_groups_simulated_then_every_row_carries_position_points_goals(self):
        """
        Each row in every one of the 12 groups must carry position, points, and
        goalsFor — the fields the downstream bracket-slotter uses to rank third-placed teams.
        """
        group_labels = [f"GROUP_{c}" for c in "ABCDEFGHIJKL"]
        played = [LiveMatch(label, f"H_{label}", f"A_{label}", 1, 0) for label in group_labels]
        result, _ = _run(played=played)
        for label in group_labels:
            for row in result[label]:
                assert "position" in row, f"'position' missing in {label} row {row}"
                assert "points" in row, f"'points' missing in {label} row {row}"
                assert "goalsFor" in row, f"'goalsFor' missing in {label} row {row}"


# ===========================================================================
# Criterion 4 (Issue #44): state.standings seeds the table; no re-derivation
# ===========================================================================


class TestStandingsSeeding:
    """
    When TournamentState.standings carries pre-scored rows, the simulator must use
    them as the starting table instead of deriving standings from played_matches.
    """

    def test_when_state_standings_provided_then_seeded_points_carry_into_final_output(self):
        """
        Provide state.standings with Spain at 3pts but NO played_matches for the group.
        One remaining fixture produces a 0-0 draw (+1pt for each team).
        If seeding works: Spain final = 4pts (3 seeded + 1 draw).
        If re-derived from scratch: Spain final = 1pt (only the draw, no history).
        """
        seeded_standings = {
            "SEED_A": [
                {
                    "team": {"name": "Spain"},
                    "position": 1,
                    "points": 3,
                    "goalsFor": 2,
                    "goalsAgainst": 0,
                    "goalDifference": 2,
                    "playedGames": 1,
                    "won": 1,
                    "draw": 0,
                    "lost": 0,
                },
                {
                    "team": {"name": "Morocco"},
                    "position": 2,
                    "points": 0,
                    "goalsFor": 0,
                    "goalsAgainst": 2,
                    "goalDifference": -2,
                    "playedGames": 1,
                    "won": 0,
                    "draw": 0,
                    "lost": 1,
                },
            ]
        }
        # No played_matches — standings must come entirely from the seed.
        # Remaining draw: Spain vs Morocco 0-0 → each gets +1pt.
        result, _ = _run(
            played=(),
            remaining=(GroupFixture("SEED_A", "Spain", "Morocco"),),
            scorelines=((0, 0),),
            initial_standings=seeded_standings,
        )
        spain = next(e for e in result["SEED_A"] if e["team"]["name"] == "Spain")
        assert spain["points"] == 4, (
            f"Spain must have 4pts (3 seeded + 1 draw); got {spain['points']}. "
            "If the simulator ignored state.standings and re-derived from played_matches "
            "(which are empty), Spain would only have 1pt — this test proves seeding works."
        )

    def test_when_state_standings_provided_with_no_remaining_fixtures_then_seeded_rows_are_returned(
        self,
    ):
        """
        When the group is already complete (no remaining fixtures), the seeded
        standings should flow through unchanged — every seeded team must appear in
        the output with the same points and goal tallies.
        """
        seeded_standings = {
            "SEED_B": [
                {
                    "team": {"name": "France"},
                    "position": 1,
                    "points": 7,
                    "goalsFor": 5,
                    "goalsAgainst": 1,
                    "goalDifference": 4,
                    "playedGames": 3,
                    "won": 2,
                    "draw": 1,
                    "lost": 0,
                },
                {
                    "team": {"name": "Australia"},
                    "position": 2,
                    "points": 4,
                    "goalsFor": 3,
                    "goalsAgainst": 3,
                    "goalDifference": 0,
                    "playedGames": 3,
                    "won": 1,
                    "draw": 1,
                    "lost": 1,
                },
                {
                    "team": {"name": "Denmark"},
                    "position": 3,
                    "points": 2,
                    "goalsFor": 2,
                    "goalsAgainst": 3,
                    "goalDifference": -1,
                    "playedGames": 3,
                    "won": 0,
                    "draw": 2,
                    "lost": 1,
                },
                {
                    "team": {"name": "Tunisia"},
                    "position": 4,
                    "points": 1,
                    "goalsFor": 1,
                    "goalsAgainst": 4,
                    "goalDifference": -3,
                    "playedGames": 3,
                    "won": 0,
                    "draw": 1,
                    "lost": 2,
                },
            ]
        }
        result, _ = _run(
            played=(),
            remaining=(),
            initial_standings=seeded_standings,
        )
        assert "SEED_B" in result, "Seeded group must appear in output even with no matches"
        by_name = {e["team"]["name"]: e for e in result["SEED_B"]}
        assert by_name["France"]["points"] == 7
        assert by_name["Australia"]["points"] == 4
        assert by_name["Denmark"]["points"] == 2
        assert by_name["Tunisia"]["points"] == 1

    def test_when_seeded_standings_and_played_matches_coexist_then_played_matches_are_not_double_counted(
        self,
    ):
        """
        Issue #44 edge case: if played_matches are ALSO present alongside state.standings,
        the simulator must not double-count the played results.
        Strategy: provide one played match whose contribution is already baked into
        state.standings. The match adds 3pts if counted once, 6pts if double-counted.
        """
        # Spain beat Morocco 2-0 — already encoded in state.standings (Spain 3pts, Morocco 0pts).
        played = [LiveMatch("SEED_C", "Spain", "Morocco", 2, 0)]
        seeded = {
            "SEED_C": [
                {
                    "team": {"name": "Spain"},
                    "position": 1,
                    "points": 3,
                    "goalsFor": 2,
                    "goalsAgainst": 0,
                    "goalDifference": 2,
                    "playedGames": 1,
                    "won": 1,
                    "draw": 0,
                    "lost": 0,
                },
                {
                    "team": {"name": "Morocco"},
                    "position": 2,
                    "points": 0,
                    "goalsFor": 0,
                    "goalsAgainst": 2,
                    "goalDifference": -2,
                    "playedGames": 1,
                    "won": 0,
                    "draw": 0,
                    "lost": 1,
                },
            ]
        }
        result, _ = _run(
            played=played,
            remaining=(),
            initial_standings=seeded,
        )
        spain = next(e for e in result["SEED_C"] if e["team"]["name"] == "Spain")
        assert spain["points"] == 3, (
            f"Spain must have exactly 3pts (the match is already in the seed); "
            f"got {spain['points']} — double-counting played_matches when state.standings "
            "is present would inflate this to 6pts."
        )


# ===========================================================================
# Criterion 5: Deterministic given a seed
# ===========================================================================


class TestDeterminism:
    def test_when_same_seed_used_twice_then_identical_standings_are_produced(self):
        """Identical TournamentState + same seed must yield byte-for-byte identical output."""
        remaining = [
            GroupFixture("D1", "Alpha", "Beta"),
            GroupFixture("D1", "Gamma", "Delta"),
            GroupFixture("D1", "Alpha", "Gamma"),
            GroupFixture("D1", "Beta", "Delta"),
            GroupFixture("D1", "Alpha", "Delta"),
            GroupFixture("D1", "Beta", "Gamma"),
        ]
        result1, _ = _run(remaining=remaining, scorelines=((1, 0),), seed=42)
        result2, _ = _run(remaining=remaining, scorelines=((1, 0),), seed=42)
        assert result1 == result2, "Same seed + same state must produce identical standings"

    def test_when_different_seeds_used_on_unbreakable_tie_then_outcomes_can_differ(self):
        """
        Different seeds applied to a coin-flip scenario must be able to produce
        different orderings — proving the seed actually drives the outcome.
        """
        played = [
            LiveMatch("D2", "Red", "Blue", 1, 0),
            LiveMatch("D2", "Blue", "Green", 1, 0),
            LiveMatch("D2", "Green", "Red", 1, 0),
        ]
        state = TournamentState(played_matches=played, remaining_group_fixtures=[])
        first_place_teams = set()
        for seed in range(20):
            spy = ScorelineSamplerSpy()
            rng = random.Random(seed)
            result = GroupStageSimulator(sampler=spy, rng=rng).simulate(state)
            first = next(e["team"]["name"] for e in result["D2"] if e["position"] == 1)
            first_place_teams.add(first)

        assert len(first_place_teams) > 1, (
            "With 20 different seeds on an unbreakable 3-way tie, at least 2 distinct "
            "first-place teams must appear — otherwise the seed is not influencing the result"
        )


# ===========================================================================
# Property-based tests (Hypothesis)
# Invariants derived from criteria text, not from any implementation source.
# ===========================================================================


@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_when_any_integer_seed_is_used_then_same_seed_always_yields_identical_output(seed):
    """
    Property: for ALL integer seeds, two runs with the same seed and identical
    state produce identical output.
    Invariant (from 'Deterministic given a seed'): determinism holds domain-wide.
    """
    played = [
        LiveMatch("PROP_A", "Alpha", "Beta", 2, 0),
        LiveMatch("PROP_A", "Alpha", "Gamma", 1, 1),
        LiveMatch("PROP_A", "Beta", "Gamma", 0, 3),
    ]
    state = TournamentState(played_matches=played, remaining_group_fixtures=[])

    r1 = GroupStageSimulator(sampler=ScorelineSamplerSpy(), rng=random.Random(seed)).simulate(state)
    r2 = GroupStageSimulator(sampler=ScorelineSamplerSpy(), rng=random.Random(seed)).simulate(state)

    assert r1 == r2, f"Seed {seed}: identical inputs must yield identical output"


@given(
    scorelines=st.lists(
        st.tuples(st.integers(0, 5), st.integers(0, 5)),
        min_size=3,
        max_size=3,
    )
)
def test_when_any_scorelines_are_used_then_positions_are_always_contiguous_from_one(scorelines):
    """
    Property: for ANY scorelines in a 3-team round-robin, the resulting positions
    are always exactly [1, 2, 3] — ranking always produces a total order.
    Invariant (from 'Ranks each group'): position set = {1..N} with no gaps.
    """
    pairs = [("Alpha", "Beta"), ("Alpha", "Gamma"), ("Beta", "Gamma")]
    played = [LiveMatch("PROP_B", h, a, hg, ag) for (h, a), (hg, ag) in zip(pairs, scorelines)]
    state = TournamentState(played_matches=played, remaining_group_fixtures=[])
    standings = GroupStageSimulator(sampler=ScorelineSamplerSpy(), rng=random.Random(0)).simulate(
        state
    )

    positions = sorted(e["position"] for e in standings["PROP_B"])
    assert positions == [1, 2, 3], (
        f"Positions must always be [1, 2, 3]; got {positions} for scorelines {scorelines}"
    )


@given(n_remaining=st.integers(min_value=0, max_value=6))
def test_when_n_remaining_fixtures_given_then_sampler_is_called_exactly_n_times(n_remaining):
    """
    Property: the number of ScorelineSampler.sample() calls equals exactly the
    number of remaining_group_fixtures for all valid non-negative counts.
    Invariant (from criterion 1): sampler is called once-per-fixture, no more, no less.
    """
    # Use distinct team names per fixture to avoid ambiguity in the group registry.
    remaining = [GroupFixture("PROP_C", f"Home{i}", f"Away{i}") for i in range(n_remaining)]
    state = TournamentState(played_matches=[], remaining_group_fixtures=remaining)
    spy = ScorelineSamplerSpy(scorelines=((1, 0),))
    GroupStageSimulator(sampler=spy, rng=random.Random(0)).simulate(state)

    assert len(spy.calls) == n_remaining, (
        f"Expected {n_remaining} sampler calls; got {len(spy.calls)}"
    )


@given(seeded_points=st.integers(min_value=0, max_value=9))
def test_when_any_seeded_points_value_then_draw_adds_exactly_one_point_on_top(seeded_points):
    """
    Property (Issue #44 criterion 4 — standings seeding invariant): for ANY integer
    seeded_points (0–9), if state.standings seeds TeamA with that many points and
    the single remaining fixture ends in a draw (0-0, +1pt per side), then TeamA's
    final points must be seeded_points + 1 — the seeded value is always carried forward.

    This invariant fails if the simulator re-derives standings from played_matches
    (which are empty here), producing 1pt regardless of seeded_points.
    """
    seeded_standings: dict = {
        "PROP_SEED": [
            {
                "team": {"name": "TeamA"},
                "position": 1,
                "points": seeded_points,
                "goalsFor": seeded_points,
                "goalsAgainst": 0,
                "goalDifference": seeded_points,
                "playedGames": seeded_points // 3,
                "won": seeded_points // 3,
                "draw": 0,
                "lost": 0,
            },
            {
                "team": {"name": "TeamB"},
                "position": 2,
                "points": 0,
                "goalsFor": 0,
                "goalsAgainst": seeded_points,
                "goalDifference": -seeded_points,
                "playedGames": seeded_points // 3,
                "won": 0,
                "draw": 0,
                "lost": seeded_points // 3,
            },
        ]
    }
    state = TournamentState(
        played_matches=[],
        remaining_group_fixtures=[GroupFixture("PROP_SEED", "TeamA", "TeamB")],
        standings=seeded_standings,
    )
    spy = ScorelineSamplerSpy(scorelines=((0, 0),))  # draw → +1pt each
    result = GroupStageSimulator(sampler=spy, rng=random.Random(0)).simulate(state)

    team_a = next(e for e in result["PROP_SEED"] if e["team"]["name"] == "TeamA")
    expected = seeded_points + 1
    assert team_a["points"] == expected, (
        f"Seeded {seeded_points}pts + draw(1pt) must yield {expected}pts; "
        f"got {team_a['points']} — seeded_points={seeded_points}"
    )
