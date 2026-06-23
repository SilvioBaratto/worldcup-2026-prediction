"""Tests for ResultPlotter — bracket and probability visualizations.

Uses the Agg non-interactive backend so tests run headlessly.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # Must be set before importing pyplot

from pathlib import Path


from worldcup_playoff.config import BracketConfig, Matchup, VisualizationConfig
from worldcup_playoff.simulation.tournament import (
    RoundResult,
    build_bracket_tree,
    extract_bracket_slots,
)
from worldcup_playoff.visualization.plots import ResultPlotter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_4_matchup_bracket() -> list[Matchup]:
    return [
        Matchup(home="Brazil", away="France", group="A"),
        Matchup(home="Germany", away="Argentina", group="B"),
        Matchup(home="Spain", away="England", group="C"),
        Matchup(home="Portugal", away="Netherlands", group="D"),
    ]


ALL_TEAMS = [
    "Brazil",
    "France",
    "Germany",
    "Argentina",
    "Spain",
    "England",
    "Portugal",
    "Netherlands",
]


def _make_round_results(
    teams: list[str], n_rounds: int, n_sim: int = 100
) -> dict[int, RoundResult]:
    """Build realistic RoundResult objects for each round."""
    rounds: dict[int, RoundResult] = {}
    for rnd in range(n_rounds):
        counts = {t: n_sim // len(teams) for t in teams}
        rounds[rnd] = RoundResult(counts=counts, n_simulations=n_sim)
    return rounds


# ---------------------------------------------------------------------------
# build_bracket_tree and extract_bracket_slots
# ---------------------------------------------------------------------------


class TestBuildBracketTree:
    def test_root_has_all_teams(self) -> None:
        root = build_bracket_tree(_make_4_matchup_bracket())
        assert set(root.teams) == set(ALL_TEAMS)

    def test_leaf_count_matches_matchup_count(self) -> None:
        root = build_bracket_tree(_make_4_matchup_bracket())
        slots = extract_bracket_slots(root)
        assert len(slots[0]) == 4

    def test_two_matchup_tree_structure(self) -> None:
        matchups = [
            Matchup(home="Brazil", away="France"),
            Matchup(home="Germany", away="Argentina"),
        ]
        root = build_bracket_tree(matchups)
        assert set(root.teams) == {"Brazil", "France", "Germany", "Argentina"}
        assert root.children is not None
        assert root.children[0].teams == ["Brazil", "France"]
        assert root.children[1].teams == ["Germany", "Argentina"]


class TestExtractBracketSlots:
    def test_round_count_for_four_matchups(self) -> None:
        """4 matchups -> 3 rounds (0, 1, 2)."""
        root = build_bracket_tree(_make_4_matchup_bracket())
        slots = extract_bracket_slots(root)
        assert set(slots.keys()) == {0, 1, 2}

    def test_slot_counts_per_round(self) -> None:
        root = build_bracket_tree(_make_4_matchup_bracket())
        slots = extract_bracket_slots(root)
        assert len(slots[0]) == 4
        assert len(slots[1]) == 2
        assert len(slots[2]) == 1

    def test_two_matchup_round_count(self) -> None:
        matchups = [
            Matchup(home="Brazil", away="France"),
            Matchup(home="Germany", away="Argentina"),
        ]
        root = build_bracket_tree(matchups)
        slots = extract_bracket_slots(root)
        assert set(slots.keys()) == {0, 1}
        assert len(slots[0]) == 2
        assert len(slots[1]) == 1


# ---------------------------------------------------------------------------
# Group propagation
# ---------------------------------------------------------------------------


class TestGroupPropagation:
    def test_leaf_slots_carry_group(self) -> None:
        matchups = _make_4_matchup_bracket()
        root = build_bracket_tree(matchups)
        slots = extract_bracket_slots(root)
        # Each leaf's group field must match its source Matchup.group
        leaf_groups = [s.group for s in slots[0]]
        assert leaf_groups == ["A", "B", "C", "D"]

    def test_root_has_empty_group_when_groups_differ(self) -> None:
        matchups = _make_4_matchup_bracket()
        root = build_bracket_tree(matchups)
        assert root.group == ""


# ---------------------------------------------------------------------------
# ResultPlotter.plot_bracket
# ---------------------------------------------------------------------------


class TestPlotBracket:
    def test_smoke_saves_png_file(self, tmp_path: Path) -> None:
        matchups = _make_4_matchup_bracket()
        bracket_cfg = BracketConfig(name="Test WC 2026", matchups=matchups)
        viz_cfg = VisualizationConfig(dpi=50)

        rounds = _make_round_results(ALL_TEAMS, n_rounds=3)
        plotter = ResultPlotter(viz_cfg)
        out = tmp_path / "bracket.png"
        plotter.plot_bracket(rounds, bracket_cfg, output_path=out)

        assert out.exists()
        assert out.stat().st_size > 0

    def test_plot_bracket_with_minimal_bracket(self, tmp_path: Path) -> None:
        matchups = [
            Matchup(home="Brazil", away="France"),
            Matchup(home="Germany", away="Argentina"),
        ]
        bracket_cfg = BracketConfig(name="Mini", matchups=matchups)
        viz_cfg = VisualizationConfig(dpi=50)

        rounds: dict[int, RoundResult] = {
            0: RoundResult(
                counts={"Brazil": 5, "France": 5, "Germany": 7, "Argentina": 3}, n_simulations=10
            ),
            1: RoundResult(
                counts={"Brazil": 6, "France": 4, "Germany": 6, "Argentina": 4}, n_simulations=10
            ),
        }
        plotter = ResultPlotter(viz_cfg)
        out = tmp_path / "mini_bracket.png"
        plotter.plot_bracket(rounds, bracket_cfg, output_path=out)

        assert out.exists()
        assert out.stat().st_size > 0

    def test_plot_bracket_empty_matchups_does_not_crash(self, tmp_path: Path) -> None:
        """Empty matchup list should be handled gracefully (early return)."""
        bracket_cfg = BracketConfig(name="Empty", matchups=[])
        viz_cfg = VisualizationConfig(dpi=50)
        rounds: dict[int, RoundResult] = {}
        plotter = ResultPlotter(viz_cfg)
        out = tmp_path / "empty.png"
        plotter.plot_bracket(rounds, bracket_cfg, output_path=out)
        # File may not be created for empty bracket; no crash is the guarantee


# ---------------------------------------------------------------------------
# ResultPlotter.plot_round_probabilities
# ---------------------------------------------------------------------------


class TestPlotRoundProbabilities:
    def test_smoke_saves_png_file(self, tmp_path: Path) -> None:
        viz_cfg = VisualizationConfig(dpi=50)
        teams = ["Brazil", "France", "Germany", "Argentina"]
        rounds = _make_round_results(teams, n_rounds=3, n_sim=100)

        plotter = ResultPlotter(viz_cfg)
        out = tmp_path / "probabilities.png"
        plotter.plot_round_probabilities(rounds, output_path=out)

        assert out.exists()
        assert out.stat().st_size > 0

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        viz_cfg = VisualizationConfig(dpi=50)
        teams = ["Brazil", "France"]
        rounds = _make_round_results(teams, n_rounds=2, n_sim=10)

        plotter = ResultPlotter(viz_cfg)
        out = tmp_path / "deep" / "nested" / "probs.png"
        plotter.plot_round_probabilities(rounds, output_path=out)

        assert out.exists()
