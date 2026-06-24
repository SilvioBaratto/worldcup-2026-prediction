"""
Tests for visualization/forecast_plots.py  —  Issue #49.

SOURCE-BLIND: no implementation source was read.  All tests are derived from the
acceptance criteria only (Red phase of TDD).

Prior contracts (from the Issue #23 iteration) are kept intact; new tests for
Issue #49 are appended.

Contracts / design choices recorded here
-----------------------------------------
- ``plot_title_odds(forecast_result, output_path, *, config=None) -> Path``
    Writes a horizontal-bar PNG; the team with the highest probability is at the
    **visual top** of the chart.  The result object exposes
    ``.champion_probabilities: dict[str, float]`` or is itself a plain dict.

- ``plot_round_advancement(round_probs, output_path, *, config=None) -> Path``
    Writes a heatmap PNG; ``round_probs`` is ``{round_name: {team: prob}}``.
    **Columns must follow ``WC_ROUND_ORDER`` intersected with the rounds present**
    regardless of input key order.

- ``WC_ROUND_ORDER`` — ordered tuple/list of canonical WC round names exported
    from ``worldcup_playoff.visualization.forecast_plots``.

- Both functions live in ``worldcup_playoff.visualization.forecast_plots``
    (``ResultPlotter`` continues to live in ``worldcup_playoff.visualization.plots``).

- Both functions use / tolerate the Agg (headless) backend.

- ``VisualizationConfig`` is imported from ``worldcup_playoff.config``; fields
    include ``dpi`` (int), ``style`` (str), and ``output_dir`` (str | Path).
"""

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib
import matplotlib.figure

matplotlib.use("Agg")  # must be set before pyplot import; no display

import matplotlib.pyplot as plt
import pytest
from hypothesis import given, settings, strategies as st

from worldcup_playoff.visualization.forecast_plots import (
    WC_ROUND_ORDER,
    plot_round_advancement,
    plot_title_odds,
)
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
    """Minimal synthetic object duck-typing ForecastResult."""
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


# ── WC_ROUND_ORDER sanity ─────────────────────────────────────────────────────
#
# Issue #49 adds WC_ROUND_ORDER as a canonical ordering constant.


def test_wc_round_order_is_a_non_empty_sequence_of_strings():
    """WC_ROUND_ORDER must be exported and contain at least one non-empty string."""
    assert len(WC_ROUND_ORDER) >= 1
    assert all(isinstance(r, str) and r for r in WC_ROUND_ORDER)


# ── Heatmap column order follows WC_ROUND_ORDER (the explicit Issue #49 requirement)


@pytest.fixture()
def capturing_savefig(monkeypatch):
    """Spy on Figure.savefig; collect each Figure object at save time."""
    captured: list[matplotlib.figure.Figure] = []
    original = matplotlib.figure.Figure.savefig

    def _spy(self, *args, **kwargs):
        captured.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", _spy)
    return captured


def _xtick_texts(fig: matplotlib.figure.Figure) -> list[str]:
    """Force a draw and return non-empty x-tick label strings from the first axes."""
    ax = fig.axes[0]
    fig.canvas.draw()
    return [t.get_text() for t in ax.get_xticklabels() if t.get_text()]


def test_when_rounds_in_reverse_canonical_order_heatmap_columns_follow_wc_round_order(
    tmp_path, capturing_savefig
):
    """Heatmap columns must be reordered to WC_ROUND_ORDER regardless of input key order.

    This is the explicit assertion called out in criterion 5 of Issue #49.
    """
    present = list(WC_ROUND_ORDER)[:4]
    # Deliberately supply rounds in reverse order so the function MUST reorder.
    reversed_rounds = list(reversed(present))
    round_probs = {r: {"Brazil": 0.5, "France": 0.4} for r in reversed_rounds}

    output = tmp_path / "colorder.png"
    plot_round_advancement(round_probs, output)

    assert capturing_savefig, "plot_round_advancement must save at least one figure"
    x_labels = _xtick_texts(capturing_savefig[-1])

    expected = [r for r in WC_ROUND_ORDER if r in set(present)]
    assert x_labels == expected, (
        f"Expected column order {expected} (WC_ROUND_ORDER).\n"
        f"Got: {x_labels}\n"
        f"Input was in reversed order: {reversed_rounds}"
    )


def test_when_subset_of_rounds_present_absent_rounds_are_excluded_from_columns(
    tmp_path, capturing_savefig
):
    """Only rounds present in the data should appear as columns (intersect semantics)."""
    first = list(WC_ROUND_ORDER)[0]
    last = list(WC_ROUND_ORDER)[-1]
    round_probs = {
        first: {"Brazil": 1.0, "France": 0.9},
        last: {"Brazil": 0.1, "France": 0.05},
    }
    output = tmp_path / "subset_rounds.png"
    plot_round_advancement(round_probs, output)

    x_labels = _xtick_texts(capturing_savefig[-1])
    assert x_labels == [first, last], f"Expected exactly [{first!r}, {last!r}], got {x_labels}"


# ── VisualizationConfig: dpi is respected ────────────────────────────────────


def test_when_higher_dpi_configured_output_png_is_larger(tmp_path):
    """Higher DPI → more pixels → larger file (config.dpi must be respected)."""
    lo_dir = tmp_path / "lo"
    hi_dir = tmp_path / "hi"
    lo_dir.mkdir()
    hi_dir.mkdir()

    path_lo = tmp_path / "lo" / "title.png"
    path_hi = tmp_path / "hi" / "title.png"

    cfg_lo = VisualizationConfig(dpi=50)
    cfg_hi = VisualizationConfig(dpi=200)

    plot_title_odds(_make_forecast_result(), path_lo, config=cfg_lo)
    plot_title_odds(_make_forecast_result(), path_hi, config=cfg_hi)

    assert path_hi.stat().st_size > path_lo.stat().st_size, (
        "DPI 200 PNG should be larger in bytes than DPI 50 PNG"
    )


# ── Property-based tests ──────────────────────────────────────────────────────
#
# 1. Both functions must be total (never-raise) over their stated domains.
# 2. Column order must follow WC_ROUND_ORDER for ANY non-empty subset of rounds.


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
    """
    round_probs = {r: {"Team A": 0.5, "Team B": 0.5} for r in round_keys}
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "prop_round_adv.png"
        plot_round_advancement(round_probs, output)


@given(
    st.lists(
        st.sampled_from(list(WC_ROUND_ORDER)),
        min_size=1,
        max_size=len(list(WC_ROUND_ORDER)),
        unique=True,
    )
)
@settings(max_examples=25, deadline=None)
def test_when_any_wc_round_subset_provided_heatmap_columns_follow_canonical_order(
    rounds: list[str],
) -> None:
    """Ordering invariant: for any non-empty subset of WC_ROUND_ORDER, the heatmap
    columns must follow the canonical WC_ROUND_ORDER (intersected with present rounds),
    regardless of input key order.
    """
    round_probs = {r: {"Brazil": 0.5, "France": 0.4} for r in rounds}

    captured: list[matplotlib.figure.Figure] = []
    original = matplotlib.figure.Figure.savefig

    def _spy(self, *args, **kwargs):
        captured.append(self)
        return original(self, *args, **kwargs)

    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "prop_colorder.png"
        with patch.object(matplotlib.figure.Figure, "savefig", _spy):
            plot_round_advancement(round_probs, output)

    if not captured:
        return  # other tests verify the file-creation contract

    fig = captured[-1]
    ax = fig.axes[0]
    fig.canvas.draw()

    x_labels = [t.get_text() for t in ax.get_xticklabels() if t.get_text()]
    expected = [r for r in WC_ROUND_ORDER if r in set(rounds)]

    assert x_labels == expected, (
        f"Input rounds={rounds!r}.\nExpected columns: {expected}\nGot: {x_labels}"
    )
