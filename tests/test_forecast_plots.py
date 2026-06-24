"""
Tests for title-odds and per-round-advancement visualizations (Issue #23).

All assertions are derived directly from the acceptance criteria.

Design choices:
- `forecast_result` duck-types `ForecastResult` from `simulation/live_forecast.py`.
  It carries `.champion_probabilities: dict[str, float]` (title odds) and
  `.round_probabilities: dict[str, dict[str, float]]` (per-round advancement).
- The per-round function `plot_round_advancement` accepts the
  `{round: probabilities}` contract as `round_probs: dict[str, dict[str, float]]`
  directly.
- Both functions accept an optional `config: VisualizationConfig` keyword arg.
- `VisualizationConfig` lives in `worldcup_playoff.config`.
"""

import tempfile
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")  # must be set before pyplot import; no display

import matplotlib.pyplot as plt
from hypothesis import given, settings, strategies as st

from worldcup_playoff.visualization.plots import plot_round_advancement, plot_title_odds
from worldcup_playoff.config import VisualizationConfig

# ── synthetic fixtures ────────────────────────────────────────────────────────

_48_TEAMS = [
    "Argentina",
    "France",
    "Brazil",
    "England",
    "Spain",
    "Germany",
    "Portugal",
    "Netherlands",
    "Belgium",
    "Uruguay",
    "Croatia",
    "Denmark",
    "Morocco",
    "USA",
    "Mexico",
    "Canada",
    "Japan",
    "South Korea",
    "Australia",
    "Senegal",
    "Nigeria",
    "Cameroon",
    "Ghana",
    "Ivory Coast",
    "Ecuador",
    "Chile",
    "Colombia",
    "Peru",
    "Switzerland",
    "Poland",
    "Serbia",
    "Hungary",
    "Czech Republic",
    "Austria",
    "Wales",
    "Scotland",
    "Ukraine",
    "Romania",
    "Turkey",
    "Saudi Arabia",
    "Iran",
    "Qatar",
    "Egypt",
    "Algeria",
    "New Zealand",
    "Panama",
    "Costa Rica",
    "Honduras",
]
assert len(_48_TEAMS) == 48, "fixture must contain exactly 48 teams"

_ROUNDS = [
    "Round of 32",
    "Round of 16",
    "Quarter-finals",
    "Semi-finals",
    "Final",
]


def _uniform_title_odds(n: int = 48) -> dict[str, float]:
    """Uniform champion probability across the first n teams."""
    return {team: 1.0 / n for team in _48_TEAMS[:n]}


def _uniform_round_probs() -> dict[str, dict[str, float]]:
    """Uniform per-round advancement probabilities for all 48 teams."""
    return {r: {team: 1.0 / 48 for team in _48_TEAMS} for r in _ROUNDS}


def _make_forecast_result(**overrides) -> SimpleNamespace:
    """Minimal synthetic object duck-typing ForecastResult (champion_probabilities / round_probabilities)."""
    defaults: dict = {
        "champion_probabilities": _uniform_title_odds(),
        "round_probabilities": _uniform_round_probs(),
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ── Criterion 1: plot_title_odds writes a PNG ─────────────────────────────────


class TestPlotTitleOdds:
    """AC: plot_title_odds(forecast_result, output_path) renders a ranked
    horizontal bar chart of champion probabilities for all 48 teams and writes a
    PNG."""

    def test_when_forecast_result_given_then_png_is_created(self, tmp_path):
        output = tmp_path / "title_odds.png"
        plot_title_odds(_make_forecast_result(), output)
        assert output.exists(), "plot_title_odds did not create the PNG file"

    def test_when_forecast_result_given_then_png_is_non_empty(self, tmp_path):
        output = tmp_path / "title_odds.png"
        plot_title_odds(_make_forecast_result(), output)
        assert output.stat().st_size > 0, "PNG file exists but is empty"

    def test_when_48_teams_in_forecast_then_function_accepts_all(self, tmp_path):
        """The spec names 48 teams; the function must handle a full 48-team dict."""
        output = tmp_path / "title_odds_48.png"
        plot_title_odds(
            _make_forecast_result(champion_probabilities=_uniform_title_odds(48)), output
        )
        assert output.exists()

    def test_when_plot_returns_then_no_matplotlib_figure_remains_open(self, tmp_path):
        """AC: figures are closed (matplotlib Agg, no display)."""
        plt.close("all")
        before = set(plt.get_fignums())
        plot_title_odds(_make_forecast_result(), tmp_path / "title_odds.png")
        after = set(plt.get_fignums())
        assert after == before, f"plot_title_odds leaked open figures: {after - before}"

    def test_when_visualization_config_supplied_then_png_is_still_written(self, tmp_path):
        """AC: reuses VisualizationConfig style/dpi/output-dir handling."""
        output = tmp_path / "title_odds_cfg.png"
        config = VisualizationConfig()
        plot_title_odds(_make_forecast_result(), output, config=config)
        assert output.exists()

    def test_when_output_path_is_string_then_png_is_created(self, tmp_path):
        """output_path may be passed as a str; the function must accept it."""
        output = str(tmp_path / "title_odds_str.png")
        plot_title_odds(_make_forecast_result(), output)
        assert Path(output).exists()


# ── Criterion 2: per-round advancement writes a PNG ───────────────────────────


class TestPlotRoundAdvancement:
    """AC: a per-round advancement visualization renders from the
    {round: probabilities} contract and writes a PNG."""

    def test_when_round_probs_given_then_png_is_created(self, tmp_path):
        output = tmp_path / "round_adv.png"
        plot_round_advancement(_uniform_round_probs(), output)
        assert output.exists(), "plot_round_advancement did not create the PNG file"

    def test_when_round_probs_given_then_png_is_non_empty(self, tmp_path):
        output = tmp_path / "round_adv.png"
        plot_round_advancement(_uniform_round_probs(), output)
        assert output.stat().st_size > 0, "PNG file exists but is empty"

    def test_when_dict_contract_supplied_then_function_accepts_it(self, tmp_path):
        """AC: '{round: probabilities}' contract — a mapping of round-name
        strings to a mapping of team-name strings to float probabilities."""
        round_probs = {
            "Round of 32": {"Team A": 0.5, "Team B": 0.5},
            "Final": {"Team A": 0.6, "Team B": 0.4},
        }
        output = tmp_path / "minimal_rounds.png"
        plot_round_advancement(round_probs, output)
        assert output.exists()

    def test_when_plot_returns_then_no_matplotlib_figure_remains_open(self, tmp_path):
        """AC: figures are closed (matplotlib Agg, no display)."""
        plt.close("all")
        before = set(plt.get_fignums())
        plot_round_advancement(_uniform_round_probs(), tmp_path / "round_adv.png")
        after = set(plt.get_fignums())
        assert after == before, f"plot_round_advancement leaked open figures: {after - before}"

    def test_when_visualization_config_supplied_then_png_is_still_written(self, tmp_path):
        """AC: reuses VisualizationConfig style/dpi/output-dir handling."""
        output = tmp_path / "round_adv_cfg.png"
        config = VisualizationConfig()
        plot_round_advancement(_uniform_round_probs(), output, config=config)
        assert output.exists()

    def test_when_output_path_is_string_then_png_is_created(self, tmp_path):
        """output_path may be passed as a str; the function must accept it."""
        output = str(tmp_path / "round_adv_str.png")
        plot_round_advancement(_uniform_round_probs(), output)
        assert Path(output).exists()


# ── Criterion 3 (combined): VisualizationConfig reuse + no figure leak ────────


class TestNoFigureAccumulation:
    """AC: Reuses the existing VisualizationConfig; figures are closed
    (matplotlib Agg, no display)."""

    def test_when_both_functions_called_repeatedly_then_no_figures_accumulate(self, tmp_path):
        """Calling both plot functions multiple times must leave zero open figures."""
        plt.close("all")
        for i in range(4):
            plot_title_odds(_make_forecast_result(), tmp_path / f"to_{i}.png")
            plot_round_advancement(_uniform_round_probs(), tmp_path / f"ra_{i}.png")
        remaining = plt.get_fignums()
        assert remaining == [], f"Open figures accumulated after repeated calls: {remaining}"


# ── Property-based tests ──────────────────────────────────────────────────────
#
# Both functions must be total (never-raise) over their stated domains.
# Invariant derived from AC: "champion probabilities for all 48 teams" and
# "renders from the {round: probabilities} contract" — any valid input in the
# stated domain must be accepted without an exception.


@given(
    title_odds=st.dictionaries(
        keys=st.text(
            min_size=1,
            max_size=30,
            alphabet=st.characters(whitelist_categories=("L", "Nd", "Zs")),
        ),
        values=st.floats(
            min_value=0.0,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False,
        ),
        min_size=1,
        max_size=48,
    )
)
@settings(max_examples=30, deadline=None)
def test_when_any_valid_title_odds_provided_then_plot_title_odds_does_not_raise(
    title_odds: dict[str, float],
) -> None:
    """Never-raises invariant: plot_title_odds is total over any non-empty dict
    of string→float probabilities in [0, 1].

    Derived from the criterion: 'champion probabilities for all 48 teams' implies
    the function must handle any valid distribution, not just the exact 48-team
    fixture used in example tests.
    """
    forecast = SimpleNamespace(champion_probabilities=title_odds, round_probabilities={})
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "prop_title_odds.png"
        plot_title_odds(forecast, output)


@given(
    round_keys=st.lists(
        st.text(min_size=1, max_size=20),
        min_size=1,
        max_size=6,
        unique=True,
    )
)
@settings(max_examples=20, deadline=None)
def test_when_any_valid_round_probs_provided_then_plot_round_advancement_does_not_raise(
    round_keys: list[str],
) -> None:
    """Never-raises invariant: plot_round_advancement is total over any
    non-empty mapping of round-name strings to probability dicts.

    Derived from the criterion: 'renders from the {round: probabilities} contract'
    implies the function must accept any non-empty round-name→probs mapping.
    """
    round_probs = {r: {"Team A": 0.5, "Team B": 0.5} for r in round_keys}
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "prop_round_adv.png"
        plot_round_advancement(round_probs, output)
