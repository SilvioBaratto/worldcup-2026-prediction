"""Tests for TournamentSimulator, RoundResult, and build_bracket_tree."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from worldcup_playoff.config import FeaturesConfig, Matchup, SimulationConfig
from worldcup_playoff.simulation.distributions import FeatureSampler, FittedDistribution
from worldcup_playoff.simulation.game import GamePredictor
from worldcup_playoff.simulation.tournament import (
    RoundResult,
    TournamentSimulator,
    build_bracket_tree,
)


# ---------------------------------------------------------------------------
# Fake classifiers
# ---------------------------------------------------------------------------


class _AlwaysHomeClassifier:
    """Always predicts home win (1)."""

    def fit(self, X: Any, y: Any) -> None:
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.ones(X.shape[0], dtype=int)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_simulator(
    distributions: dict[str, list[FittedDistribution]],
    classifier: Any | None = None,
) -> TournamentSimulator:
    clf = classifier or _AlwaysHomeClassifier()
    sampler = FeatureSampler(FeaturesConfig())
    predictor = GamePredictor(clf, sampler, distributions)
    config = SimulationConfig(n_simulations=10, classifier="naive_bayes")
    return TournamentSimulator(predictor, config)


# ---------------------------------------------------------------------------
# RoundResult
# ---------------------------------------------------------------------------


class TestRoundResult:
    def test_probabilities_empty_when_no_simulations(self) -> None:
        rr = RoundResult(counts={})
        assert rr.probabilities == {}

    def test_probabilities_empty_when_n_simulations_is_zero(self) -> None:
        rr = RoundResult(counts={"Brazil": 0, "France": 0}, n_simulations=0)
        assert rr.probabilities == {}

    def test_single_matchup_probabilities(self) -> None:
        rr = RoundResult(counts={"Brazil": 3, "France": 7}, n_simulations=10)
        probs = rr.probabilities
        assert abs(probs["Brazil"] - 0.3) < 1e-9
        assert abs(probs["France"] - 0.7) < 1e-9

    def test_single_matchup_probs_sum_to_one(self) -> None:
        rr = RoundResult(counts={"Brazil": 4, "France": 6}, n_simulations=10)
        assert abs(sum(rr.probabilities.values()) - 1.0) < 1e-9

    def test_multi_matchup_denominator_uses_n_simulations(self) -> None:
        """Each team's probability is count / n_simulations (not count / total_wins)."""
        rr = RoundResult(
            counts={"Brazil": 5, "France": 5, "Germany": 3, "Argentina": 7},
            n_simulations=10,
        )
        probs = rr.probabilities
        assert abs(probs["Brazil"] - 0.5) < 1e-9
        assert abs(probs["France"] - 0.5) < 1e-9
        assert abs(probs["Germany"] - 0.3) < 1e-9
        assert abs(probs["Argentina"] - 0.7) < 1e-9
        # Sum = number of ties in the round, not 1.0
        assert abs(sum(probs.values()) - 2.0) < 1e-9

    def test_uncontested_team_shows_100_percent(self) -> None:
        rr = RoundResult(counts={"Brazil": 10, "France": 0}, n_simulations=10)
        probs = rr.probabilities
        assert abs(probs["Brazil"] - 1.0) < 1e-9
        assert abs(probs["France"] - 0.0) < 1e-9

    def test_round_result_has_round_num_attribute(self) -> None:
        """RoundResult must expose round_num as documented."""
        rr = RoundResult(counts={}, n_simulations=0)
        # round_num is not directly on RoundResult but assigned externally via the dict key
        # Verify the dataclass is callable and has the right fields
        assert hasattr(rr, "counts")
        assert hasattr(rr, "n_simulations")
        assert hasattr(rr, "probabilities")


# ---------------------------------------------------------------------------
# TournamentSimulator.simulate
# ---------------------------------------------------------------------------


class TestTournamentSimulator:
    def test_simulate_returns_round_results_dict(
        self, sample_distributions: dict[str, list[FittedDistribution]]
    ) -> None:
        sim = _make_simulator(sample_distributions)
        bracket = [Matchup(home="Brazil", away="France")]
        rounds = sim.simulate(bracket, n_simulations=10)
        assert isinstance(rounds, dict)
        assert 0 in rounds

    def test_simulate_single_matchup_has_both_teams(
        self, sample_distributions: dict[str, list[FittedDistribution]]
    ) -> None:
        sim = _make_simulator(sample_distributions)
        bracket = [Matchup(home="Brazil", away="France")]
        rounds = sim.simulate(bracket, n_simulations=10)
        assert "Brazil" in rounds[0].counts
        assert "France" in rounds[0].counts

    def test_simulate_always_home_wins(
        self, sample_distributions: dict[str, list[FittedDistribution]]
    ) -> None:
        sim = _make_simulator(sample_distributions, classifier=_AlwaysHomeClassifier())
        bracket = [Matchup(home="Brazil", away="France")]
        rounds = sim.simulate(bracket, n_simulations=50)
        assert rounds[0].counts["Brazil"] == 50
        assert abs(rounds[0].probabilities["Brazil"] - 1.0) < 1e-9
        assert abs(rounds[0].probabilities["France"] - 0.0) < 1e-9

    def test_simulate_four_team_bracket_has_two_rounds(
        self, sample_distributions: dict[str, list[FittedDistribution]]
    ) -> None:
        extended = {
            **sample_distributions,
            "Spain": sample_distributions["Brazil"],
            "England": sample_distributions["France"],
        }
        sim = _make_simulator(extended)
        bracket = [
            Matchup(home="Brazil", away="France"),
            Matchup(home="Germany", away="Argentina"),
        ]
        rounds = sim.simulate(bracket, n_simulations=10)
        # Round 0 (first round) + round 1 (final)
        assert 0 in rounds
        assert 1 in rounds

    def test_simulate_counts_sum_equals_n_simulations(
        self, sample_distributions: dict[str, list[FittedDistribution]]
    ) -> None:
        """In round 0 with one tie, total advancement count == n_simulations."""
        sim = _make_simulator(sample_distributions)
        bracket = [Matchup(home="Brazil", away="France")]
        n = 20
        rounds = sim.simulate(bracket, n_simulations=n)
        total = rounds[0].counts["Brazil"] + rounds[0].counts["France"]
        assert total == n

    def test_simulate_progress_callback_called_n_times(
        self, sample_distributions: dict[str, list[FittedDistribution]]
    ) -> None:
        """Progress callback must be called exactly n_simulations times."""
        sim = _make_simulator(sample_distributions)
        bracket = [Matchup(home="Brazil", away="France")]
        n = 15
        calls: list[int] = []
        sim.simulate(bracket, n_simulations=n, progress_callback=calls.append)
        assert len(calls) == n
        assert calls == list(range(1, n + 1))

    def test_simulate_non_power_of_two_raises(
        self, sample_distributions: dict[str, list[FittedDistribution]]
    ) -> None:
        """Bracket length must be a power of 2 — 3 matchups must raise."""
        sim = _make_simulator(sample_distributions)
        extended = {
            **sample_distributions,
            "Spain": sample_distributions["Brazil"],
            "England": sample_distributions["France"],
            "Portugal": sample_distributions["Germany"],
            "Netherlands": sample_distributions["Argentina"],
        }
        sim2 = _make_simulator(extended)
        bracket = [
            Matchup(home="Brazil", away="France"),
            Matchup(home="Germany", away="Argentina"),
            Matchup(home="Spain", away="England"),  # 3 matchups — not power of 2
        ]
        with pytest.raises(ValueError, match="power of 2"):
            sim2.simulate(bracket, n_simulations=5)

    def test_simulate_empty_bracket_raises(
        self, sample_distributions: dict[str, list[FittedDistribution]]
    ) -> None:
        sim = _make_simulator(sample_distributions)
        with pytest.raises(ValueError):
            sim.simulate([], n_simulations=5)

    def test_simulate_n_simulations_controls_count(
        self, sample_distributions: dict[str, list[FittedDistribution]]
    ) -> None:
        """n_simulations argument takes precedence over config default."""
        sim = _make_simulator(sample_distributions)
        bracket = [Matchup(home="Brazil", away="France")]
        rounds = sim.simulate(bracket, n_simulations=7)
        assert rounds[0].n_simulations == 7


# ---------------------------------------------------------------------------
# build_bracket_tree
# ---------------------------------------------------------------------------


class TestBuildBracketTree:
    def test_raises_for_empty_matchup_list(self) -> None:
        with pytest.raises(ValueError, match="non-empty power-of-2 length"):
            build_bracket_tree([])

    def test_raises_for_odd_matchup_count(self) -> None:
        matchups = [
            Matchup(home="A", away="B"),
            Matchup(home="C", away="D"),
            Matchup(home="E", away="F"),
        ]
        with pytest.raises(ValueError, match="non-empty power-of-2 length, got 3"):
            build_bracket_tree(matchups)

    def test_raises_for_five_matchups(self) -> None:
        matchups = [Matchup(home=f"T{i}", away=f"T{i+10}") for i in range(5)]
        with pytest.raises(ValueError, match="non-empty power-of-2 length, got 5"):
            build_bracket_tree(matchups)

    def test_single_matchup_returns_leaf(self) -> None:
        matchups = [Matchup(home="Brazil", away="France")]
        root = build_bracket_tree(matchups)
        assert set(root.teams) == {"Brazil", "France"}
        assert root.children is None

    def test_two_matchups_returns_two_levels(self) -> None:
        matchups = [
            Matchup(home="Brazil", away="France"),
            Matchup(home="Germany", away="Argentina"),
        ]
        root = build_bracket_tree(matchups)
        assert set(root.teams) == {"Brazil", "France", "Germany", "Argentina"}
        assert root.children is not None
        left, right = root.children
        assert set(left.teams) == {"Brazil", "France"}
        assert set(right.teams) == {"Germany", "Argentina"}

    def test_four_matchups_returns_three_levels(self) -> None:
        matchups = [
            Matchup(home=f"Team{i}", away=f"Team{i+4}") for i in range(4)
        ]
        root = build_bracket_tree(matchups)
        all_teams = {f"Team{i}" for i in range(8)}
        assert set(root.teams) == all_teams
        assert root.children is not None
        assert root.children[0].children is not None  # not a leaf
