"""Title-odds and per-round advancement visualizations for the live forecast.

Implements two public plotting functions:
- ``plot_title_odds``: ranked horizontal bar chart of champion probabilities.
- ``plot_round_advancement``: heatmap of per-round advancement fractions.

Both duck-type against ``ForecastResult`` (``champion_probabilities``,
``round_probabilities``) and the ``{round: {team: prob}}`` contract.
Figures are always closed after saving — never displayed (Agg backend).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np

from worldcup_playoff.config import VisualizationConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _resolve_config(config: VisualizationConfig | None) -> VisualizationConfig:
    return config or VisualizationConfig()


def _apply_style(config: VisualizationConfig) -> None:
    plt.style.use(config.style)
    plt.rcParams["figure.dpi"] = config.dpi


def _save_and_close(fig: plt.Figure, output_path: Path | str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    logger.info("Plot saved to %s", path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Title-odds: ranked horizontal bar chart
# ---------------------------------------------------------------------------


def _sorted_champion_probs(forecast_result: Any) -> list[tuple[str, float]]:
    return sorted(
        forecast_result.champion_probabilities.items(),
        key=lambda kv: kv[1],
        reverse=True,
    )


def _build_title_odds_axes(
    ax: matplotlib.axes.Axes,
    items: list[tuple[str, float]],
) -> None:
    """Render a ranked horizontal bar chart onto *ax*."""
    teams = [t for t, _ in items]
    values = [p for _, p in items]
    n = len(teams)
    ax.barh(range(n), values, align="center", color="steelblue", edgecolor="white")
    ax.set_yticks(range(n))
    ax.set_yticklabels(teams, fontsize=max(5, 9 - n // 10))
    ax.invert_yaxis()
    ax.set_xlabel("Champion Probability")
    ax.set_title("FIFA World Cup 2026 — Title Odds")


def _build_title_odds_figure(items: list[tuple[str, float]]) -> plt.Figure:
    n = len(items)
    fig, ax = plt.subplots(figsize=(10, max(6, n * 0.28)))
    _build_title_odds_axes(ax, items)
    plt.tight_layout()
    return fig


def plot_title_odds(
    forecast_result: Any,
    output_path: Path | str,
    config: VisualizationConfig | None = None,
) -> None:
    """Render a ranked horizontal bar chart of champion probabilities and save a PNG.

    Reads ``forecast_result.champion_probabilities`` duck-typed.

    Args:
        forecast_result: Object exposing ``.champion_probabilities: dict[str, float]``.
        output_path: Destination PNG path (``str`` or ``Path``).
        config: Visualization settings; uses :class:`VisualizationConfig` defaults if None.
    """
    cfg = _resolve_config(config)
    _apply_style(cfg)
    fig = _build_title_odds_figure(_sorted_champion_probs(forecast_result))
    _save_and_close(fig, output_path)


# ---------------------------------------------------------------------------
# Per-round advancement: heatmap
# ---------------------------------------------------------------------------


def _round_team_matrix(
    rounds: list[str],
    teams: list[str],
    round_probs: dict[str, dict[str, float]],
) -> np.ndarray:
    return np.array([[round_probs[r].get(t, 0.0) for r in rounds] for t in teams])


def _configure_heatmap(
    ax: matplotlib.axes.Axes,
    data: np.ndarray,
    rounds: list[str],
    teams: list[str],
) -> Any:
    """Paint the heatmap image and configure axis labels; returns the image."""
    im = ax.imshow(data, aspect="auto", cmap="Blues")
    ax.set_xticks(range(len(rounds)))
    ax.set_xticklabels(rounds, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(teams)))
    ax.set_yticklabels(teams, fontsize=max(4, 8 - len(teams) // 10))
    return im


def _build_round_advancement_figure(
    round_probs: dict[str, dict[str, float]],
) -> plt.Figure:
    rounds = list(round_probs.keys())
    teams = sorted({t for probs in round_probs.values() for t in probs})
    fig, ax = plt.subplots(figsize=(max(6, len(rounds) * 1.5), max(6, len(teams) * 0.28)))
    if not rounds or not teams:
        return fig
    data = _round_team_matrix(rounds, teams, round_probs)
    im = _configure_heatmap(ax, data, rounds, teams)
    fig.colorbar(im, ax=ax, label="Advancement Probability")
    ax.set_title("FIFA World Cup 2026 — Per-Round Advancement Probabilities")
    plt.tight_layout()
    return fig


def plot_round_advancement(
    round_probs: dict[str, dict[str, float]],
    output_path: Path | str,
    config: VisualizationConfig | None = None,
) -> None:
    """Render a per-round advancement heatmap and save a PNG.

    Accepts the ``{round_name: {team: probability}}`` contract directly.

    Args:
        round_probs: Mapping of round name to ``{team: advancement_probability}``.
        output_path: Destination PNG path (``str`` or ``Path``).
        config: Visualization settings; uses :class:`VisualizationConfig` defaults if None.
    """
    cfg = _resolve_config(config)
    _apply_style(cfg)
    fig = _build_round_advancement_figure(round_probs)
    _save_and_close(fig, output_path)
