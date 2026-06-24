"""
Source-blind tests for worldcup_playoff.simulation.knockout — Issues #18 and #45.

Authored from acceptance criteria ONLY; no implementation source has been read.
All tests are in the Red phase of TDD and will fail until the module is implemented.

Criteria covered (issue #18):
  [UNIT] resolve_r32(standings_dict) produces 16 ties matching R32_SLOTS.
  [UNIT] Single tie: regulation → extra-time (scaled by extra_time_factor) → penalty coin-flip.
  [UNIT] Seed reproducibility for the penalty coin-flip.

Criteria covered (issue #45):
  [UNIT] resolve_r32(standings) yields exactly 16 concrete (home, away) ties from a v4 standings dict
  [UNIT] simulate folds R32→R16→QF→SF→Final and returns per-round RoundResult advancement counts
  [UNIT] Two simulate calls with the same seed produce identical counts (determinism)

Criteria skipped (NOT VERIFIABLE per oracle):
  All tests pass — boilerplate suite gate; no per-criterion assertion.
  SOLID, clean code — subjective code-quality prose; no concrete runtime assertion.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

# ──────────────────────────────────────────────────────────────────────────────
# Test helpers
# ──────────────────────────────────────────────────────────────────────────────


def _stub_sampler(sequence: list[tuple[int, int]]):
    """
    Return a callable sampler that yields scorelines from *sequence* in order.

    Raises StopIteration when exhausted — any test that over-calls has a bug.
    Signature mirrors the expected production interface: sampler(home, away) → (h_goals, a_goals).
    """
    it = iter(sequence)

    def _call(home: str, away: str) -> tuple[int, int]:
        return next(it)

    return _call


def _make_standings(groups_teams: dict[str, list[str]]) -> dict:
    """
    Build a minimal standings dict from {group_letter: [team1, team2, team3, team4]}
    where teams are already ordered 1st→4th with synthetic, consistent stats.
    """
    standings: dict = {}
    for group, teams in groups_teams.items():
        standings[group] = [
            {
                "position": rank,
                "team": {"name": team},
                "points": max(9 - (rank - 1) * 3, 0),
                "goalsFor": max(5 - rank, 0),
                "goalsAgainst": rank - 1,
                "goalDifference": max(6 - 2 * rank, -2),
            }
            for rank, team in enumerate(teams, start=1)
        ]
    return standings


# Synthetic WC2026 setup: 12 groups (A–L), 4 teams each.
# Teams named "{Group}{Position}" so assertions are self-documenting.
_GROUPS: dict[str, list[str]] = {g: [f"{g}1", f"{g}2", f"{g}3", f"{g}4"] for g in "ABCDEFGHIJKL"}
_STANDINGS = _make_standings(_GROUPS)

_FOURTH_PLACED = {f"{g}4" for g in "ABCDEFGHIJKL"}
_TOP_TWO = {f"{g}{p}" for g in "ABCDEFGHIJKL" for p in (1, 2)}
_THIRDS = {f"{g}3" for g in "ABCDEFGHIJKL"}  # 12 thirds; 8 qualify


# ──────────────────────────────────────────────────────────────────────────────
# Criterion: resolve_r32 pairings match R32_SLOTS for given standings
# ──────────────────────────────────────────────────────────────────────────────


def test_when_12_group_standings_given_then_resolve_r32_returns_16_ties():
    """resolve_r32 must produce exactly 16 ties for the Round of 32."""
    from worldcup_playoff.simulation.knockout import resolve_r32

    ties = resolve_r32(_STANDINGS)

    assert len(ties) == 16


def test_when_standings_given_then_each_r32_tie_is_a_pair_of_nonempty_team_names():
    """Every element of the resolve_r32 result must be a (home, away) pair of non-empty strings."""
    from worldcup_playoff.simulation.knockout import resolve_r32

    ties = resolve_r32(_STANDINGS)

    for pair in ties:
        home, away = pair
        assert isinstance(home, str) and home, f"home name invalid: {home!r}"
        assert isinstance(away, str) and away, f"away name invalid: {away!r}"


def test_when_standings_given_then_no_team_appears_twice_across_r32_ties():
    """No team may appear in more than one R32 tie simultaneously."""
    from worldcup_playoff.simulation.knockout import resolve_r32

    ties = resolve_r32(_STANDINGS)

    all_teams = [team for pair in ties for team in pair]
    assert len(all_teams) == len(set(all_teams)), "A team appears in more than one R32 tie"


def test_when_standings_given_then_fourth_placed_teams_are_excluded_from_r32():
    """
    4th-placed finishers do not qualify for the Round of 32 under WC2026 rules
    (top-2 of each group + 8 best third-placed = 32 qualifiers).
    """
    from worldcup_playoff.simulation.knockout import resolve_r32

    ties = resolve_r32(_STANDINGS)

    for home, away in ties:
        assert home not in _FOURTH_PLACED, f"4th-placed team {home!r} must not reach R32"
        assert away not in _FOURTH_PLACED, f"4th-placed team {away!r} must not reach R32"


def test_when_r32_slots_exported_then_it_has_exactly_16_entries():
    """
    R32_SLOTS must be a module-level constant exporting exactly 16 bracket-slot entries,
    one per Round-of-32 tie, used by resolve_r32 to pair teams from group standings.
    """
    from worldcup_playoff.simulation.knockout import R32_SLOTS

    assert len(R32_SLOTS) == 16


def test_when_group_a_positions_swapped_then_r32_pairings_change():
    """
    resolve_r32 reads actual standings, not hard-coded names.
    Swapping group-A's winner and runner-up must alter at least one R32 matchup.
    """
    from worldcup_playoff.simulation.knockout import resolve_r32

    alt_groups = dict(_GROUPS)
    alt_groups["A"] = ["A2", "A1", "A3", "A4"]  # winner ↔ runner-up
    alt_standings = _make_standings(alt_groups)

    original_ties = [tuple(t) for t in resolve_r32(_STANDINGS)]
    swapped_ties = [tuple(t) for t in resolve_r32(alt_standings)]

    assert original_ties != swapped_ties, (
        "Swapping group-A standings must change at least one R32 pairing"
    )


def test_when_same_standings_called_twice_then_resolve_r32_is_pure():
    """
    resolve_r32 is a pure function — identical input always yields identical output
    (no hidden random state in bracket construction itself).
    """
    from worldcup_playoff.simulation.knockout import resolve_r32

    assert resolve_r32(_STANDINGS) == resolve_r32(_STANDINGS)


# ──────────────────────────────────────────────────────────────────────────────
# Criterion: single tie resolution (regulation → ET → penalty coin-flip)
# ──────────────────────────────────────────────────────────────────────────────


def test_when_home_wins_in_regulation_then_home_team_is_returned():
    """Regulation home win: home team advances; extra-time path must NOT be triggered."""
    from worldcup_playoff.simulation.knockout import resolve_tie

    winner = resolve_tie(
        "Brazil",
        "France",
        sampler=_stub_sampler([(2, 0)]),
        extra_time_factor=0.33,
        seed=0,
    )

    assert winner == "Brazil"


def test_when_away_wins_in_regulation_then_away_team_is_returned():
    """Regulation away win: away team advances."""
    from worldcup_playoff.simulation.knockout import resolve_tie

    winner = resolve_tie(
        "Brazil",
        "France",
        sampler=_stub_sampler([(0, 3)]),
        extra_time_factor=0.33,
        seed=0,
    )

    assert winner == "France"


def test_when_regulation_draw_and_away_scores_in_extra_time_then_away_advances():
    """
    Regulation draw triggers extra-time resampling.
    Away goal in ET (0 home, 1 away) → away team advances.
    """
    from worldcup_playoff.simulation.knockout import resolve_tie

    winner = resolve_tie(
        "Brazil",
        "France",
        sampler=_stub_sampler([(1, 1), (0, 1)]),
        extra_time_factor=0.33,
        seed=0,
    )

    assert winner == "France"


def test_when_regulation_draw_and_home_scores_in_extra_time_then_home_advances():
    """Regulation draw → home goal in ET → home team advances."""
    from worldcup_playoff.simulation.knockout import resolve_tie

    winner = resolve_tie(
        "Brazil",
        "France",
        sampler=_stub_sampler([(0, 0), (1, 0)]),
        extra_time_factor=0.33,
        seed=0,
    )

    assert winner == "Brazil"


def test_when_draw_after_extra_time_then_penalty_coin_flip_decides():
    """
    Regulation draw AND extra-time draw → penalty coin-flip decides.
    Winner must be one of the two competing teams.
    """
    from worldcup_playoff.simulation.knockout import resolve_tie

    winner = resolve_tie(
        "Brazil",
        "France",
        sampler=_stub_sampler([(1, 1), (0, 0)]),
        extra_time_factor=0.33,
        seed=42,
    )

    assert winner in {"Brazil", "France"}


def test_when_draw_after_extra_time_then_penalty_winner_is_deterministic_given_seed():
    """
    Seeded penalty coin-flip: identical arguments including seed → identical winner
    across repeated independent calls.
    """
    from worldcup_playoff.simulation.knockout import resolve_tie

    results = []
    for _ in range(5):
        w = resolve_tie(
            "Brazil",
            "France",
            sampler=_stub_sampler([(0, 0), (0, 0)]),
            extra_time_factor=0.33,
            seed=99,
        )
        results.append(w)

    assert len(set(results)) == 1, (
        "Penalty coin-flip with a fixed seed must always return the same winner"
    )


def test_when_different_seeds_used_then_both_teams_can_win_on_penalties():
    """
    The coin-flip genuinely uses the seed: across varied seeds, both teams win at least once.
    This rules out a constant implementation that always returns the same team.
    """
    from worldcup_playoff.simulation.knockout import resolve_tie

    outcomes: set[str] = set()
    for seed in range(20):
        w = resolve_tie(
            "Brazil",
            "France",
            sampler=_stub_sampler([(0, 0), (0, 0)]),
            extra_time_factor=0.33,
            seed=seed,
        )
        outcomes.add(w)

    assert outcomes == {"Brazil", "France"}, (
        "Coin-flip over 20 seeds must produce both team outcomes"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Criterion 6: seed reproducibility (explicitly listed as unit-verifiable)
# ──────────────────────────────────────────────────────────────────────────────


def test_when_same_home_away_and_seed_then_penalty_result_is_reproducible_across_calls():
    """
    resolve_tie with identical (home, away, seed) is reproducible — meets the criterion
    'identical standings + seed → identical champion' at the single-tie level.
    """
    from worldcup_playoff.simulation.knockout import resolve_tie

    first = resolve_tie(
        "Germany",
        "Argentina",
        sampler=_stub_sampler([(2, 2), (0, 0)]),
        extra_time_factor=0.33,
        seed=7,
    )
    second = resolve_tie(
        "Germany",
        "Argentina",
        sampler=_stub_sampler([(2, 2), (0, 0)]),
        extra_time_factor=0.33,
        seed=7,
    )

    assert first == second


# ──────────────────────────────────────────────────────────────────────────────
# Property-based tests — issue #18 invariants
# Invariants derived from criterion text — not from any implementation.
# ──────────────────────────────────────────────────────────────────────────────


@given(
    home_goals=st.integers(min_value=0, max_value=10),
    away_goals=st.integers(min_value=0, max_value=10),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_when_any_non_draw_scoreline_then_resolve_tie_returns_one_of_the_two_teams(
    home_goals: int,
    away_goals: int,
    seed: int,
) -> None:
    """
    Totality invariant (never-raises-for-valid-input):
    resolve_tie accepts any non-draw scoreline and always returns exactly one of {home, away}.
    Draw path requires ET stub and is excluded here to keep the example self-contained.
    """
    from worldcup_playoff.simulation.knockout import resolve_tie

    if home_goals == away_goals:
        return  # draw path requires ET resampling; tested in dedicated example tests

    winner = resolve_tie(
        "Alpha",
        "Beta",
        sampler=_stub_sampler([(home_goals, away_goals)]),
        extra_time_factor=0.33,
        seed=seed,
    )

    assert winner in {"Alpha", "Beta"}


@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_when_penalty_path_then_same_seed_produces_same_winner_idempotently(seed: int) -> None:
    """
    Idempotence / seed-stability invariant:
    resolve_tie(... seed=s) == resolve_tie(... seed=s) for any seed s
    when both regulation and extra-time end level.
    This encodes 'deterministic given a seed' from the criterion.
    """
    from worldcup_playoff.simulation.knockout import resolve_tie

    w1 = resolve_tie(
        "Alpha",
        "Beta",
        sampler=_stub_sampler([(0, 0), (0, 0)]),
        extra_time_factor=0.33,
        seed=seed,
    )
    w2 = resolve_tie(
        "Alpha",
        "Beta",
        sampler=_stub_sampler([(0, 0), (0, 0)]),
        extra_time_factor=0.33,
        seed=seed,
    )

    assert w1 == w2


# ══════════════════════════════════════════════════════════════════════════════
# Issue #45 additions
# Criteria: simulate folds all 5 rounds; determinism across two calls; new
# determinism test (criterion 5: "tests/test_knockout.py passes plus a new
# determinism test").
# ══════════════════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────────────────────
# Stateless samplers for simulate (unlimited calls; no StopIteration risk)
# ──────────────────────────────────────────────────────────────────────────────


def _always_home_wins(home: str, away: str) -> tuple[int, int]:
    """Home always wins 2-0; no ET or penalty path is triggered."""
    return (2, 0)


def _always_draw_sampler(home: str, away: str) -> tuple[int, int]:
    """Every match draws 1-1, forcing ET → penalty coin-flip in every tie."""
    return (1, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Criterion 3 (issue #45): simulate folds R32→R16→QF→SF→Final
# ──────────────────────────────────────────────────────────────────────────────


def test_when_simulate_called_then_exactly_five_round_results_are_returned():
    """simulate must return one RoundResult per round: R32, R16, QF, SF, Final."""
    from worldcup_playoff.simulation.knockout import resolve_r32, simulate

    ties = resolve_r32(_STANDINGS)
    results = simulate(ties, _always_home_wins, seed=0, n_simulations=10)

    assert len(results) == 5


def test_when_simulate_called_then_every_round_result_has_a_probabilities_dict():
    from worldcup_playoff.simulation.knockout import resolve_r32, simulate

    ties = resolve_r32(_STANDINGS)
    for rr in simulate(ties, _always_home_wins, seed=0, n_simulations=10):
        assert hasattr(rr, "probabilities") and isinstance(rr.probabilities, dict)


def test_when_simulate_called_then_r32_result_has_exactly_16_advancing_teams():
    """Round of 32: 16 ties → 16 winners."""
    from worldcup_playoff.simulation.knockout import resolve_r32, simulate

    ties = resolve_r32(_STANDINGS)
    results = simulate(ties, _always_home_wins, seed=0, n_simulations=10)

    assert len(results[0].probabilities) == 16


def test_when_simulate_called_then_round_advancing_team_counts_are_16_8_4_2_1():
    """R32→R16→QF→SF→Final halves the field each round until one champion."""
    from worldcup_playoff.simulation.knockout import resolve_r32, simulate

    ties = resolve_r32(_STANDINGS)
    results = simulate(ties, _always_home_wins, seed=0, n_simulations=10)

    assert [len(rr.probabilities) for rr in results] == [16, 8, 4, 2, 1]


def test_when_simulate_called_then_final_probabilities_sum_equals_n_simulations():
    """Every simulation produces exactly one champion → final counts must sum to n_simulations."""
    from worldcup_playoff.simulation.knockout import resolve_r32, simulate

    n = 20
    ties = resolve_r32(_STANDINGS)
    results = simulate(ties, _always_home_wins, seed=0, n_simulations=n)

    assert sum(results[-1].probabilities.values()) == n


def test_when_simulate_called_then_all_probability_values_are_non_negative():
    from worldcup_playoff.simulation.knockout import resolve_r32, simulate

    ties = resolve_r32(_STANDINGS)
    for rr in simulate(ties, _always_home_wins, seed=0, n_simulations=10):
        assert all(v >= 0 for v in rr.probabilities.values())


def test_when_simulate_called_then_r16_teams_are_a_subset_of_r32_advancing_teams():
    """Bracket propagation: only R32 winners may appear in R16 — no phantom teams."""
    from worldcup_playoff.simulation.knockout import resolve_r32, simulate

    ties = resolve_r32(_STANDINGS)
    results = simulate(ties, _always_home_wins, seed=0, n_simulations=10)
    r32_teams = set(results[0].probabilities)
    r16_teams = set(results[1].probabilities)

    assert r16_teams.issubset(r32_teams)


def test_when_simulate_called_then_final_teams_are_a_subset_of_sf_advancing_teams():
    """Only SF winners may appear in the Final."""
    from worldcup_playoff.simulation.knockout import resolve_r32, simulate

    ties = resolve_r32(_STANDINGS)
    results = simulate(ties, _always_home_wins, seed=0, n_simulations=10)
    sf_teams = set(results[3].probabilities)
    final_teams = set(results[4].probabilities)

    assert final_teams.issubset(sf_teams)


# ──────────────────────────────────────────────────────────────────────────────
# Criterion 4 + Criterion 5 determinism test (issue #45)
# "Two simulate calls with the same seed produce identical counts"
# "tests/test_knockout.py passes plus a new determinism test"
# ──────────────────────────────────────────────────────────────────────────────


def test_when_same_seed_given_then_simulate_produces_identical_counts_across_all_rounds():
    """
    New determinism test (criterion 5 + criterion 4):
    simulate(ties, sampler, seed=s, n_simulations=N) called twice with the same seed
    must return identical RoundResult.probabilities for every round — no cross-round
    seed collision may occur.
    """
    from worldcup_playoff.simulation.knockout import resolve_r32, simulate

    ties = resolve_r32(_STANDINGS)
    run_a = simulate(ties, _always_home_wins, seed=42, n_simulations=50)
    run_b = simulate(ties, _always_home_wins, seed=42, n_simulations=50)

    for i, (ra, rb) in enumerate(zip(run_a, run_b)):
        assert ra.probabilities == rb.probabilities, (
            f"Round {i}: same seed=42 must produce identical advancement counts"
        )


def test_when_same_seed_given_with_penalty_path_then_simulate_is_still_deterministic():
    """
    Determinism must hold even when penalty coin-flips are triggered in every tie
    (_always_draw_sampler forces ET → penalty for every match in the bracket).
    A seed collision between rounds would surface here first.
    """
    from worldcup_playoff.simulation.knockout import resolve_r32, simulate

    ties = resolve_r32(_STANDINGS)
    run_a = simulate(ties, _always_draw_sampler, seed=99, n_simulations=50)
    run_b = simulate(ties, _always_draw_sampler, seed=99, n_simulations=50)

    for i, (ra, rb) in enumerate(zip(run_a, run_b)):
        assert ra.probabilities == rb.probabilities, (
            f"Round {i}: penalty-heavy simulate must also be deterministic (seed=99)"
        )


# Property: determinism holds for any seed value (idempotence invariant)
@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(max_examples=25)
def test_when_simulate_called_twice_with_any_seed_then_r32_counts_always_match(seed: int) -> None:
    """
    Idempotence property: simulate(ties, sampler, seed=s) == simulate(ties, sampler, seed=s)
    for any seed s — derived from the 'determinism, no cross-round seed collision' criterion.
    """
    from worldcup_playoff.simulation.knockout import resolve_r32, simulate

    ties = resolve_r32(_STANDINGS)
    run_a = simulate(ties, _always_home_wins, seed=seed, n_simulations=5)
    run_b = simulate(ties, _always_home_wins, seed=seed, n_simulations=5)

    assert run_a[0].probabilities == run_b[0].probabilities
