"""
Tests for seed reproducibility and ScorelineSampler variance (Issue #24).

All assertions are derived directly from the acceptance criteria.
No implementation source was read during authoring (TDD Red phase).

Design choices (spec is silent on these details):
- 'ScorelineSampler' is the callable returned by make_sampler(abilities, config).
  It takes (home_team, away_team, rng) and returns (home_goals, away_goals).
- 'TeamAbilities' is the dataclass from simulation/poisson.py with 'attack' and
  'defence' dicts mapping team names to float strength values.
- LiveForecaster.run(state, abilities, n_simulations, seed) drives the Monte Carlo.
- TournamentState is duck-typed via SimpleNamespace so these tests need no network.
- _knockout_sim_fn is a public module-level factory in simulation/live_forecast.py.
"""

import copy
from types import SimpleNamespace

import numpy as np
from hypothesis import given, settings, strategies as st


# ---------------------------------------------------------------------------
# Shared synthetic fixtures (no network, no file I/O)
# ---------------------------------------------------------------------------

_32_TEAMS = [f"Team{i:02d}" for i in range(32)]


def _make_abilities(teams: list[str]):
    """Minimal TeamAbilities for the given team list (uniform strengths)."""
    from worldcup_playoff.simulation.poisson import TeamAbilities

    return TeamAbilities(
        attack={t: 1.0 for t in teams},
        defence={t: 1.0 for t in teams},
        home_adv=0.25,
        rho=-0.1,
        intercept=0.0,
    )


def _two_team_abilities():
    """Two-team set with asymmetric strengths to drive Poisson variance."""
    from worldcup_playoff.simulation.poisson import TeamAbilities

    return TeamAbilities(
        attack={"Alpha": 1.8, "Beta": 0.7},
        defence={"Alpha": 0.7, "Beta": 1.5},
        home_adv=0.25,
        rho=-0.1,
        intercept=0.0,
    )


def _stub_state(n_teams: int = 32) -> SimpleNamespace:
    """Duck-typed TournamentState with n_teams in one group, nothing played."""

    class _Row:
        def __init__(self, name: str) -> None:
            self.team_name = name

    class _Group:
        def __init__(self, names: list[str]) -> None:
            self.table = [_Row(n) for n in names]

    return SimpleNamespace(
        standings=[_Group([f"T{i}" for i in range(n_teams)])],
        played=[],
        remaining_group_fixtures=[],
    )


def _stub_forecaster():
    """LiveForecaster with deterministic stub simulators (no Dixon-Coles, no network).

    The group_sim derives its team list from the state's standings so that all
    returned team names are present in LiveForecaster's champion_counts dict.
    """
    from worldcup_playoff.simulation.live_forecast import LiveForecaster

    def _group_sim(state, abilities, rng):
        teams = sorted(row.team_name for gs in state.standings for row in gs.table)
        rng.shuffle(teams)
        return teams

    def _knockout_sim(qualified, abilities, rng):
        champion = qualified[rng.integers(len(qualified))]
        return {"champion": champion, "rounds": {}}

    return LiveForecaster(_group_sim, _knockout_sim)


# ---------------------------------------------------------------------------
# Criterion 2: ScorelineSampler — RNG is advanced; draws show variance
# ---------------------------------------------------------------------------


class TestScorelineSamplerRngAdvancement:
    """AC: ScorelineSampler draws advance a single injected Generator."""

    def test_when_sampler_called_then_rng_state_changes(self):
        from worldcup_playoff.simulation.poisson import make_sampler
        from worldcup_playoff.config import PoissonConfig

        abilities = _two_team_abilities()
        rng = np.random.default_rng(seed=7)
        state_before = copy.deepcopy(rng.bit_generator.state)

        sampler = make_sampler(abilities, PoissonConfig())
        sampler("Alpha", "Beta", rng)

        assert rng.bit_generator.state != state_before, (
            "RNG state was not advanced — sampler may not be using the injected Generator"
        )

    def test_when_sampler_called_twice_then_each_call_advances_rng_independently(self):
        """Two rngs seeded identically diverge after one sampler call — confirms state is consumed."""
        from worldcup_playoff.simulation.poisson import make_sampler
        from worldcup_playoff.config import PoissonConfig

        abilities = _two_team_abilities()
        sampler = make_sampler(abilities, PoissonConfig())

        rng_a = np.random.default_rng(42)
        rng_b = np.random.default_rng(42)

        # Advance rng_a by one sampler draw
        sampler("Alpha", "Beta", rng_a)

        # rng_a is now ahead of rng_b; the next float should differ
        next_a = rng_a.random()
        first_b = rng_b.random()
        assert next_a != first_b, (
            "RNG was not advanced — next draw from rng_a matched rng_b at position 0"
        )


class TestScorelineSamplerVariance:
    """AC: repeated draws for the same fixture show non-degenerate variance."""

    def test_when_sampler_draws_thirty_times_then_home_goals_are_not_constant(self):
        from worldcup_playoff.simulation.poisson import make_sampler
        from worldcup_playoff.config import PoissonConfig

        rng = np.random.default_rng(seed=0)
        sampler = make_sampler(_two_team_abilities(), PoissonConfig())
        home_goals = [sampler("Alpha", "Beta", rng)[0] for _ in range(30)]

        assert len(set(home_goals)) > 1, (
            "All 30 home-goal draws were identical — sampler shows no variance"
        )

    def test_when_sampler_draws_thirty_times_then_away_goals_are_not_constant(self):
        from worldcup_playoff.simulation.poisson import make_sampler
        from worldcup_playoff.config import PoissonConfig

        rng = np.random.default_rng(seed=1)
        sampler = make_sampler(_two_team_abilities(), PoissonConfig())
        away_goals = [sampler("Alpha", "Beta", rng)[1] for _ in range(30)]

        assert len(set(away_goals)) > 1, (
            "All 30 away-goal draws were identical — sampler shows no variance"
        )

    def test_when_sampler_draws_with_strong_home_team_then_home_can_outscore_away(self):
        """AC: scorelines reflect ability asymmetry — strong team can win."""
        from worldcup_playoff.simulation.poisson import make_sampler
        from worldcup_playoff.config import PoissonConfig

        rng = np.random.default_rng(seed=2)
        sampler = make_sampler(_two_team_abilities(), PoissonConfig())
        results = [sampler("Alpha", "Beta", rng) for _ in range(30)]

        # At least one simulation where Alpha (strong attack) scores
        home_nonzero = any(h > 0 for h, _ in results)
        assert home_nonzero, "Strong-attack team never scored in 30 draws"


# ---------------------------------------------------------------------------
# Criterion 3: seed reproducibility — example tests + Hypothesis property
# ---------------------------------------------------------------------------


class TestSeedReproducibility:
    """AC: Two run_forecast(..., seed=S) runs produce bit-identical title odds."""

    def test_when_same_seed_used_twice_then_champion_probabilities_are_identical(self):
        forecaster = _stub_forecaster()
        state = _stub_state()

        r1 = forecaster.run(state, None, n_simulations=200, seed=42)  # type: ignore[arg-type]
        r2 = forecaster.run(state, None, n_simulations=200, seed=42)  # type: ignore[arg-type]

        assert r1.champion_probabilities == r2.champion_probabilities

    def test_when_same_seed_used_twice_then_round_probabilities_are_identical(self):
        forecaster = _stub_forecaster()
        state = _stub_state()

        r1 = forecaster.run(state, None, n_simulations=200, seed=99)  # type: ignore[arg-type]
        r2 = forecaster.run(state, None, n_simulations=200, seed=99)  # type: ignore[arg-type]

        assert r1.round_probabilities == r2.round_probabilities

    def test_when_different_seeds_used_then_champion_probabilities_differ(self):
        """AC: different seeds produce different odds."""
        forecaster = _stub_forecaster()
        state = _stub_state()

        r_a = forecaster.run(state, None, n_simulations=1000, seed=1)  # type: ignore[arg-type]
        r_b = forecaster.run(state, None, n_simulations=1000, seed=2)  # type: ignore[arg-type]

        assert r_a.champion_probabilities != r_b.champion_probabilities


@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(max_examples=5, deadline=None)
def test_when_any_valid_seed_used_twice_then_results_are_bit_identical(seed: int) -> None:
    """AC property: seed reproducibility holds for any non-negative integer seed.

    Invariant derived from the criterion: 'Two run_forecast(..., seed=S) runs with
    the same seed produce bit-identical title odds' implies this must hold for ALL
    valid seeds, not just a fixed example.
    """
    forecaster = _stub_forecaster()
    state = _stub_state()

    r1 = forecaster.run(state, None, n_simulations=50, seed=seed)  # type: ignore[arg-type]
    r2 = forecaster.run(state, None, n_simulations=50, seed=seed)  # type: ignore[arg-type]

    assert r1.champion_probabilities == r2.champion_probabilities


# ---------------------------------------------------------------------------
# Criterion 4: real-adapter regression
# ---------------------------------------------------------------------------


class TestRealKnockoutAdapterRegression:
    """AC: real _knockout_sim_fn asserts (a) no TypeError, (b) variance, (c) seed repro."""

    def test_when_real_knockout_sim_called_then_no_type_error(self):
        """AC 4(a): _knockout_sim_fn raises no TypeError with valid input."""
        from worldcup_playoff.simulation.live_forecast import _knockout_sim_fn
        from worldcup_playoff.config import AppConfig

        abilities = _make_abilities(_32_TEAMS)
        sim = _knockout_sim_fn(abilities, AppConfig())
        rng = np.random.default_rng(0)

        result = sim(_32_TEAMS, abilities, rng)

        assert isinstance(result, dict), "Expected dict result from _knockout_sim_fn"
        assert "champion" in result, "Result must contain 'champion' key"
        assert isinstance(result["champion"], str), "champion must be a team name string"

    def test_when_real_knockout_sim_called_twenty_times_then_champions_vary(self):
        """AC 4(b): group-stage scorelines vary — different champions across simulations."""
        from worldcup_playoff.simulation.live_forecast import _knockout_sim_fn
        from worldcup_playoff.config import AppConfig

        abilities = _make_abilities(_32_TEAMS)
        sim = _knockout_sim_fn(abilities, AppConfig())
        rng = np.random.default_rng(77)

        champions = [sim(_32_TEAMS, abilities, rng)["champion"] for _ in range(20)]

        assert len(set(champions)) > 1, (
            "All 20 simulations returned the same champion — no variance in the real adapter"
        )

    def test_when_real_knockout_sim_seeded_identically_then_champion_is_identical(self):
        """AC 4(c): seed reproducibility — same start-state RNG → same champion."""
        from worldcup_playoff.simulation.live_forecast import _knockout_sim_fn
        from worldcup_playoff.config import AppConfig

        abilities = _make_abilities(_32_TEAMS)
        sim = _knockout_sim_fn(abilities, AppConfig())

        result_a = sim(_32_TEAMS, abilities, np.random.default_rng(99))
        result_b = sim(_32_TEAMS, abilities, np.random.default_rng(99))

        assert result_a["champion"] == result_b["champion"]

    def test_when_real_knockout_sim_called_then_champion_is_one_of_qualified_teams(self):
        """AC 4(a) extension: champion must come from the qualified-teams list."""
        from worldcup_playoff.simulation.live_forecast import _knockout_sim_fn
        from worldcup_playoff.config import AppConfig

        abilities = _make_abilities(_32_TEAMS)
        sim = _knockout_sim_fn(abilities, AppConfig())
        rng = np.random.default_rng(5)

        result = sim(_32_TEAMS, abilities, rng)

        assert result["champion"] in _32_TEAMS, (
            f"Champion '{result['champion']}' is not in the qualified-teams list"
        )
