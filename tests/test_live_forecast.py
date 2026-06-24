"""
Tests for simulation/live_forecast.py — Issue #19.

Stubs inject controllable group-stage and knockout simulators so all tests
are self-contained, deterministic, and network-free.  The constructor
signatures for TournamentState and TeamAbilities match the real types from
data.live and simulation.poisson respectively.
"""

from __future__ import annotations

import math

from hypothesis import given, settings, strategies as st

from worldcup_playoff.data.live import GroupStanding, TableRow, TournamentState
from worldcup_playoff.simulation.live_forecast import ForecastResult, LiveForecaster
from worldcup_playoff.simulation.poisson import TeamAbilities

# ---------------------------------------------------------------------------
# Test constants — 48 teams in 12 groups of 4 (WC2026 structure)
# ---------------------------------------------------------------------------

_NUM_GROUPS = 12
_TEAMS_PER_GROUP = 4
_GROUPS: dict[str, list[str]] = {
    chr(ord("A") + g): [f"Team{chr(ord('A') + g)}{t + 1}" for t in range(_TEAMS_PER_GROUP)]
    for g in range(_NUM_GROUPS)
}
_ALL_TEAMS: list[str] = [t for group_teams in _GROUPS.values() for t in group_teams]

assert len(_ALL_TEAMS) == 48, "Test setup requires exactly 48 teams"

_QUALIFIED_32: list[str] = _ALL_TEAMS[:32]

# ---------------------------------------------------------------------------
# Stub simulators
#
# The deterministic variants never touch rng so structural/coverage tests pass
# regardless of the rng type supplied.
# The stochastic variant calls rng.choice() — numpy.random.Generator supports
# this for a Python sequence — to exercise the seeded sub-stream path.
# ---------------------------------------------------------------------------


def _fixed_group_sim(state: TournamentState, abilities: TeamAbilities, rng) -> list[str]:
    """Deterministic: always returns the first 32 teams as qualified."""
    return _QUALIFIED_32


def _fixed_knockout_sim(qualified: list[str], abilities: TeamAbilities, rng) -> dict:
    """Deterministic: always crowns qualified[0]; returns all five WC rounds."""
    return {
        "champion": qualified[0],
        "rounds": {
            "round_of_32": {t: 1 for t in qualified},
            "round_of_16": {t: 1 for t in qualified[:16]},
            "quarter_finals": {t: 1 for t in qualified[:8]},
            "semi_finals": {t: 1 for t in qualified[:4]},
            "final": {t: 1 for t in qualified[:2]},
        },
    }


def _stochastic_knockout_sim(qualified: list[str], abilities: TeamAbilities, rng) -> dict:
    """Stochastic: picks a random champion via numpy Generator.choice()."""
    champion = str(rng.choice(qualified))
    return {"champion": champion, "rounds": {}}


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_state() -> TournamentState:
    """TournamentState with all 48 teams listed in group standings.

    TableRow.team_name is set via the v4-schema model_validator (team.name),
    so we use model_validate with the API-shaped dict instead of direct kwargs.
    """
    standings = [
        GroupStanding(
            group=label,
            stage="GROUP_STAGE",
            table=[
                TableRow.model_validate(
                    {
                        "position": i + 1,
                        "team": {"name": team},
                        "playedGames": 0,
                        "won": 0,
                        "draw": 0,
                        "lost": 0,
                        "points": 0,
                        "goalsFor": 0,
                        "goalsAgainst": 0,
                        "goalDifference": 0,
                    }
                )
                for i, team in enumerate(teams)
            ],
        )
        for label, teams in _GROUPS.items()
    ]
    return TournamentState(played=[], remaining_group_fixtures=[], standings=standings)


def _make_abilities() -> TeamAbilities:
    """Uniform TeamAbilities — equal log-scale abilities for all 48 teams."""
    neutral = {t: 0.0 for t in _ALL_TEAMS}
    return TeamAbilities(attack=neutral, defence=neutral, home_adv=0.0, rho=0.0, intercept=0.0)


def _make_forecaster(*, stochastic: bool = False) -> LiveForecaster:
    knockout_sim = _stochastic_knockout_sim if stochastic else _fixed_knockout_sim
    return LiveForecaster(group_simulator=_fixed_group_sim, knockout_simulator=knockout_sim)


def _run_forecast(n: int = 20, seed: int = 42, *, stochastic: bool = False) -> ForecastResult:
    return _make_forecaster(stochastic=stochastic).run(
        state=_make_state(),
        abilities=_make_abilities(),
        n_simulations=n,
        seed=seed,
    )


# ===========================================================================
# Criterion 2 — ForecastResult value-object shape
# ===========================================================================


class TestForecastResultShape:
    def test_when_forecast_runs_then_a_forecast_result_is_returned(self) -> None:
        result = _run_forecast(n=5)
        assert isinstance(result, ForecastResult)

    def test_when_forecast_runs_then_champion_probabilities_attribute_is_a_dict(self) -> None:
        result = _run_forecast(n=5)
        assert hasattr(result, "champion_probabilities")
        assert isinstance(result.champion_probabilities, dict)

    def test_when_forecast_runs_then_round_probabilities_attribute_is_a_dict(self) -> None:
        result = _run_forecast(n=5)
        assert hasattr(result, "round_probabilities")
        assert isinstance(result.round_probabilities, dict)

    def test_when_forecast_runs_then_round_probabilities_has_at_least_one_round(self) -> None:
        """The fixed stub returns five knockout rounds; the result must expose them."""
        result = _run_forecast(n=5)
        assert len(result.round_probabilities) >= 1


# ===========================================================================
# Criterion 7 — 48-team coverage
# ===========================================================================


class TestCoverage:
    def test_when_forecast_runs_then_champion_probabilities_has_exactly_48_entries(self) -> None:
        result = _run_forecast(n=20)
        assert len(result.champion_probabilities) == 48

    def test_when_forecast_runs_then_every_team_is_present_in_champion_probabilities(self) -> None:
        """Teams that never win (p=0.0) must still appear — 'all 48 teams present'."""
        result = _run_forecast(n=10)
        assert set(result.champion_probabilities.keys()) == set(_ALL_TEAMS)


# ===========================================================================
# Criterion 7 — probability sanity
# ===========================================================================


class TestProbabilitySanity:
    def test_when_forecast_runs_then_champion_probabilities_sum_to_one(self) -> None:
        """Exactly one champion per simulation → probabilities must sum to 1.0."""
        result = _run_forecast(n=50, seed=42, stochastic=True)
        total = sum(result.champion_probabilities.values())
        assert math.isclose(total, 1.0, abs_tol=1e-9)

    def test_when_forecast_runs_then_all_champion_probabilities_are_nonnegative(self) -> None:
        result = _run_forecast(n=20, seed=1, stochastic=True)
        assert all(p >= 0.0 for p in result.champion_probabilities.values())

    def test_when_forecast_runs_then_all_champion_probabilities_are_at_most_one(self) -> None:
        result = _run_forecast(n=20, seed=2, stochastic=True)
        assert all(p <= 1.0 for p in result.champion_probabilities.values())


# ===========================================================================
# Criterion 7 — seed reproducibility
# ===========================================================================


class TestSeedReproducibility:
    def test_when_same_seed_used_twice_then_champion_probabilities_are_identical(self) -> None:
        """Same master seed → same per-iteration sub-seeds → same champion counts."""
        result_a = _run_forecast(n=50, seed=42, stochastic=True)
        result_b = _run_forecast(n=50, seed=42, stochastic=True)
        assert result_a.champion_probabilities == result_b.champion_probabilities

    def test_when_same_seed_used_twice_then_round_probabilities_are_identical(self) -> None:
        result_a = _run_forecast(n=50, seed=99, stochastic=True)
        result_b = _run_forecast(n=50, seed=99, stochastic=True)
        assert result_a.round_probabilities == result_b.round_probabilities


# ===========================================================================
# Criterion 1 — orchestration: TeamAbilities accepted; N iterations run
# ===========================================================================


class TestOrchestration:
    def test_when_precomputed_abilities_are_passed_then_forecast_result_is_returned(self) -> None:
        """Criterion 1 accept-path: pre-fitted TeamAbilities passed directly to run()."""
        result = _make_forecaster().run(
            state=_make_state(),
            abilities=_make_abilities(),
            n_simulations=5,
            seed=0,
        )
        assert isinstance(result, ForecastResult)

    def test_when_n_simulations_is_10_then_champion_probabilities_sum_to_one(self) -> None:
        """One champion per simulation → total_prob * n == n → probs sum to 1.0."""
        result = _run_forecast(n=10, stochastic=True, seed=0)
        total = sum(result.champion_probabilities.values())
        assert math.isclose(total, 1.0, abs_tol=1e-9)

    def test_when_n_is_varied_then_champion_probabilities_always_cover_all_teams(self) -> None:
        """All 48 teams are initialised to 0.0 before accumulation regardless of n."""
        for n in (1, 5, 100):
            result = _run_forecast(n=n, stochastic=True, seed=7)
            assert len(result.champion_probabilities) == 48, (
                f"Expected 48 teams with n={n}, got {len(result.champion_probabilities)}"
            )


# ===========================================================================
# Property-based tests (Hypothesis)
# ===========================================================================


class TestProperties:
    @given(st.integers(min_value=0, max_value=2**31 - 1))
    @settings(max_examples=8)
    def test_when_same_seed_used_twice_then_identical_champion_probabilities_are_returned(
        self, seed: int
    ) -> None:
        """Idempotence: same master seed always yields bit-identical champion probabilities."""
        a = _run_forecast(n=15, seed=seed, stochastic=True)
        b = _run_forecast(n=15, seed=seed, stochastic=True)
        assert a.champion_probabilities == b.champion_probabilities

    @given(st.integers(min_value=1, max_value=30))
    @settings(max_examples=8)
    def test_when_any_positive_n_is_used_then_champion_probabilities_sum_to_one(
        self, n: int
    ) -> None:
        """Sum invariant: champion probs sum to 1.0 for any n_simulations >= 1."""
        result = _run_forecast(n=n, seed=7, stochastic=True)
        total = sum(result.champion_probabilities.values())
        assert math.isclose(total, 1.0, abs_tol=1e-9)

    @given(st.integers(min_value=1, max_value=20))
    @settings(max_examples=8)
    def test_when_any_positive_n_is_used_then_all_48_teams_are_present(self, n: int) -> None:
        """Coverage invariant: all 48 teams appear for any n_simulations >= 1."""
        result = _run_forecast(n=n, seed=3)
        assert len(result.champion_probabilities) == 48
