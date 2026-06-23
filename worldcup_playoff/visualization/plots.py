"""Visualization of knockout-round results and advancement probabilities.

Mirrors nba_playoff/visualization/plots.py adapted for a FIFA World Cup
single-elimination bracket:
- No best-of-7 series — each tie is a single match.
- No conferences — all slots flow left-to-right toward the Final.
- Teams are country-name strings.
- RoundResult is accepted duck-typed (not imported at module level) so that
  the simulation package can be developed in parallel without creating a
  circular dependency.  The contract expected from each RoundResult value:
      .round_num: int
      .probabilities: dict[str, float]   (team -> advancement probability)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.axes
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from worldcup_playoff.config import BracketConfig, Matchup, VisualizationConfig

if TYPE_CHECKING:
    # Only used for type annotations; not imported at runtime to avoid cycles.
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain constant
# ---------------------------------------------------------------------------
ROUND_NAMES: list[str] = [
    "Round of 32",
    "Round of 16",
    "Quarter-finals",
    "Semi-finals",
    "Final",
]


# ---------------------------------------------------------------------------
# Internal bracket layout helpers
# ---------------------------------------------------------------------------

def _build_round_slots(matchups: list[Matchup]) -> dict[int, list[tuple[str, str]]]:
    """Return a mapping of round index -> list of (home, away) pairs.

    Round 0 contains the seed matchups from *matchups*.  Subsequent rounds are
    placeholder pairs ("TBD", "TBD") whose count halves each time, mirroring
    the single-elimination structure.

    Args:
        matchups: First-round matchup definitions from :class:`BracketConfig`.

    Returns:
        Dict mapping round index to a list of (home, away) tuples.
    """
    slots: dict[int, list[tuple[str, str]]] = {}
    current_pairs: list[tuple[str, str]] = [(m.home, m.away) for m in matchups]
    rnd = 0
    while current_pairs:
        slots[rnd] = current_pairs
        next_count = len(current_pairs) // 2
        current_pairs = [("TBD", "TBD")] * next_count
        rnd += 1
    return slots


def _round_label(rnd: int, total_rounds: int) -> str:
    """Return a human-readable name for *rnd* given *total_rounds*.

    The mapping aligns ROUND_NAMES from the Final backwards so that the last
    round is always "Final", the penultimate is "Semi-finals", etc.

    Args:
        rnd: Zero-based round index.
        total_rounds: Total number of rounds in the bracket.

    Returns:
        A display string such as "Quarter-finals" or "Round 0".
    """
    offset = len(ROUND_NAMES) - total_rounds
    idx = offset + rnd
    if 0 <= idx < len(ROUND_NAMES):
        return ROUND_NAMES[idx]
    return f"Round {rnd}"


# ---------------------------------------------------------------------------
# Main plotter
# ---------------------------------------------------------------------------

class ResultPlotter:
    """Generates knockout bracket and advancement probability visualizations.

    Constructed from a :class:`~worldcup_playoff.config.VisualizationConfig`
    and exposes two public methods that mirror the NBA original:
    ``plot_bracket`` and ``plot_round_probabilities``.

    Args:
        config: Visualization settings (dpi, matplotlib style, output dir).
    """

    def __init__(self, config: VisualizationConfig) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plot_round_probabilities(
        self,
        rounds: dict[int, Any],
        output_path: Path | None = None,
    ) -> None:
        """Stacked area chart of advancement probabilities by team and round.

        Each column of the stacked area corresponds to one knockout round;
        the height of each coloured band is the team's probability of having
        advanced to (or through) that round, mirroring the NBA probabilities
        plot.

        Args:
            rounds: Mapping of round index to a RoundResult-like object that
                exposes ``.probabilities`` (``dict[str, float]``).
            output_path: Destination PNG path.  If *None* the figure is shown
                interactively via ``plt.show()``.
        """
        plt.style.use(self._config.style)
        plt.rcParams["figure.dpi"] = self._config.dpi

        rounds_stats = list(rounds.values())
        team_names: list[str] = list(rounds_stats[0].probabilities.keys())
        x: list[int] = list(rounds.keys())
        y: np.ndarray = np.array(
            [list(r.probabilities.values()) for r in rounds_stats]
        ).T

        c1 = sns.color_palette("tab10", n_colors=10)
        c2 = sns.color_palette("pastel", n_colors=10)
        color_map: list[Any] = list(c1) + list(c2)

        total_rounds = len(rounds)
        x_labels = [_round_label(i, total_rounds) for i in x]

        fig = plt.figure(figsize=(14, 8))
        plt.stackplot(x, y, labels=team_names, colors=color_map)
        plt.legend(bbox_to_anchor=(1.01, 1.0), loc="upper left", fontsize=10)
        plt.xticks(x, x_labels, fontsize=12, rotation=15, ha="right")
        plt.yticks(fontsize=13)
        plt.xlabel("Round", fontsize=14)
        plt.ylabel("Cumulative Advancement Probability", fontsize=14)
        plt.title(
            "World Cup 2026 — Advancement Probabilities by Team & Round",
            pad=20,
            fontsize=18,
        )
        plt.tight_layout()

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, bbox_inches="tight")
            logger.info("Probabilities plot saved to %s", output_path)
        else:
            plt.show()
        plt.close(fig)

    def plot_bracket(
        self,
        rounds: dict[int, Any],
        bracket: BracketConfig,
        output_path: Path | None = None,
    ) -> None:
        """Render the knockout bracket as a PNG with per-team probabilities.

        Each slot box shows the two competing countries and their round
        advancement probabilities drawn from *rounds*.  Connector lines
        flow left-to-right (no conference split — World Cup has no
        conferences).

        Args:
            rounds: Mapping of round index to a RoundResult-like object with
                ``.probabilities`` (``dict[str, float]``).
            bracket: Bracket definition including first-round matchups.
            output_path: Destination PNG path.  If *None* the figure is shown
                interactively.
        """
        slots_by_round = _build_round_slots(bracket.matchups)
        if not slots_by_round:
            logger.warning("plot_bracket called with empty matchup list — nothing to draw.")
            return

        max_round = max(slots_by_round)
        total_rounds = max_round + 1

        # --- Layout constants (tweaked for left-to-right single side) ---
        BOX_W: float = 3.8
        BOX_H: float = 1.6
        ROW_GAP: float = 0.7
        COL_SPACING: float = 5.2
        MARGIN_X: float = 1.0
        MARGIN_TOP: float = 3.0
        MARGIN_BOT: float = 2.5

        n_first = len(slots_by_round[0])

        bracket_h: float = n_first * BOX_H + (n_first - 1) * ROW_GAP
        fig_w: float = COL_SPACING * total_rounds + BOX_W + 2 * MARGIN_X
        fig_h: float = bracket_h + MARGIN_TOP + MARGIN_BOT

        fig, ax = plt.subplots(figsize=(fig_w * 0.55, fig_h * 0.55))
        fig.patch.set_facecolor("#F8F9FA")
        ax.set_xlim(0, fig_w)
        ax.set_ylim(0, fig_h)
        ax.set_aspect("equal")
        ax.axis("off")

        y_top: float = fig_h - MARGIN_TOP

        # --- Title ---
        title = bracket.name or "FIFA World Cup 2026 Knockout Bracket"
        ax.text(
            fig_w / 2,
            fig_h - 0.5,
            title,
            ha="center",
            va="top",
            fontsize=15,
            fontweight="bold",
            color="#0d1b2a",
            fontfamily="sans-serif",
        )

        # --- Compute box positions for each round ---
        # Round 0 positions: evenly spaced down the left column.
        positions: dict[int, list[tuple[float, float]]] = {}
        positions[0] = [
            (MARGIN_X, y_top - i * (BOX_H + ROW_GAP))
            for i in range(n_first)
        ]

        for rnd in range(1, total_rounds):
            prev = positions[rnd - 1]
            x_col = MARGIN_X + COL_SPACING * rnd
            new_pos: list[tuple[float, float]] = []
            for i in range(0, len(prev), 2):
                if i + 1 < len(prev):
                    y_mid = (prev[i][1] + prev[i + 1][1]) / 2
                else:
                    y_mid = prev[i][1]
                new_pos.append((x_col, y_mid))
            positions[rnd] = new_pos

        # --- Draw connector lines (behind boxes) ---
        for rnd in range(max_round):
            prev_pos = positions[rnd]
            next_pos = positions[rnd + 1]
            ni = 0
            for i in range(0, len(prev_pos), 2):
                c1_xy = (prev_pos[i][0] + BOX_W, prev_pos[i][1] + BOX_H / 2)
                if i + 1 < len(prev_pos):
                    c2_xy = (
                        prev_pos[i + 1][0] + BOX_W,
                        prev_pos[i + 1][1] + BOX_H / 2,
                    )
                else:
                    c2_xy = c1_xy
                if ni < len(next_pos):
                    p_xy = (next_pos[ni][0], next_pos[ni][1] + BOX_H / 2)
                    self._draw_connector(ax, p_xy, c1_xy, c2_xy)
                ni += 1

        # --- Palette cycling (no conference colours; use round tints) ---
        PALETTES: list[tuple[str, str, str]] = [
            ("#d6eaf8", "#2e86c1", "#1a5276"),  # round 0 — cool blue
            ("#d5f5e3", "#1e8449", "#145a32"),  # round 1 — green
            ("#fdebd0", "#ca6f1e", "#784212"),  # round 2 — amber
            ("#e8daef", "#7d3c98", "#4a235a"),  # round 3 — purple
            ("#fef9e7", "#d4ac0d", "#7d6608"),  # Final — gold
        ]

        def _palette_for_round(rnd: int) -> tuple[str, str, str]:
            return PALETTES[min(rnd, len(PALETTES) - 1)]

        # --- Round column labels ---
        label_y = fig_h - 1.8
        for rnd in range(total_rounds):
            if positions[rnd]:
                col_cx = positions[rnd][0][0] + BOX_W / 2
                ax.text(
                    col_cx,
                    label_y,
                    _round_label(rnd, total_rounds),
                    ha="center",
                    va="top",
                    fontsize=8,
                    color="#555555",
                    fontfamily="sans-serif",
                )

        # --- Draw slot boxes ---
        for rnd in range(total_rounds):
            slot_list = slots_by_round.get(rnd, [])
            pos_list = positions.get(rnd, [])
            probs: dict[str, float] = (
                rounds[rnd].probabilities if rnd in rounds else {}
            )
            palette = _palette_for_round(rnd)

            for (home, away), (bx, by) in zip(slot_list, pos_list):
                teams = [home, away]
                team_probs = sorted(
                    ((t, probs.get(t, 0.0)) for t in teams),
                    key=lambda tp: tp[1],
                    reverse=True,
                )
                self._draw_team_box(ax, bx, by, team_probs, palette, BOX_W, BOX_H)

        # --- Champion banner below the Final slot ---
        if max_round in rounds:
            finals_probs: dict[str, float] = rounds[max_round].probabilities
            if finals_probs:
                champion = max(finals_probs, key=lambda t: finals_probs[t])
                champ_prob = finals_probs[champion]

                fx, fy = positions[max_round][0]
                banner_cx = fx + BOX_W / 2
                banner_y = fy - 1.8

                ax.plot(
                    [banner_cx, banner_cx],
                    [fy, banner_y + 0.7],
                    color="#d4af37",
                    lw=2.5,
                    solid_capstyle="round",
                )

                bw, bh = BOX_W + 1.0, 1.2
                champ_rect = mpatches.FancyBboxPatch(
                    (banner_cx - bw / 2, banner_y - bh / 2),
                    bw,
                    bh,
                    boxstyle=mpatches.BoxStyle.Round(pad=0.15),
                    facecolor="#fef9e7",
                    edgecolor="#d4af37",
                    linewidth=2.5,
                )
                ax.add_patch(champ_rect)

                ax.text(
                    banner_cx,
                    banner_y + 0.22,
                    "W O R L D   C U P   C H A M P I O N",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="#b7950b",
                    fontweight="bold",
                    fontfamily="sans-serif",
                )
                ax.text(
                    banner_cx,
                    banner_y - 0.22,
                    f"{champion}  ({champ_prob:.1%})",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="#0d1b2a",
                    fontweight="bold",
                    fontfamily="sans-serif",
                )

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                output_path,
                bbox_inches="tight",
                dpi=max(self._config.dpi, 150),
                facecolor=fig.get_facecolor(),
                edgecolor="none",
            )
            logger.info("Bracket plot saved to %s", output_path)
        else:
            plt.show()
        plt.close(fig)

    # ------------------------------------------------------------------
    # Private drawing helpers
    # ------------------------------------------------------------------

    def _draw_team_box(
        self,
        ax: matplotlib.axes.Axes,
        x: float,
        y: float,
        teams_with_probs: list[tuple[str, float]],
        palette: tuple[str, str, str],
        box_w: float,
        box_h: float,
    ) -> None:
        """Draw a single bracket slot box at *(x, y)*.

        The box lists competing teams vertically; the leader row (highest
        probability) receives a tinted background.  A narrow accent stripe
        runs along the left edge.

        Args:
            ax: Target matplotlib axes.
            x: Left edge of the box in axes coordinates.
            y: Bottom edge of the box in axes coordinates.
            teams_with_probs: List of (team_name, probability) pairs, sorted
                descending by probability.
            palette: Three-tuple of (background, accent, dark) hex colours.
            box_w: Box width in axes units.
            box_h: Box height in axes units.
        """
        bg, accent, dark = palette

        # Outer rounded rectangle
        rect = mpatches.FancyBboxPatch(
            (x, y),
            box_w,
            box_h,
            boxstyle=mpatches.BoxStyle.Round(pad=0.08),
            facecolor="white",
            edgecolor="#cccccc",
            linewidth=0.8,
        )
        ax.add_patch(rect)

        row_h = box_h / max(len(teams_with_probs), 1)
        for i, (name, prob) in enumerate(teams_with_probs):
            ry = y + box_h - (i + 1) * row_h
            is_leader = i == 0 and prob > 0.0

            if is_leader:
                row_bg = mpatches.FancyBboxPatch(
                    (x + 0.05, ry + 0.05),
                    box_w - 0.1,
                    row_h - 0.1,
                    boxstyle=mpatches.BoxStyle.Round(pad=0.04),
                    facecolor=bg,
                    edgecolor="none",
                )
                ax.add_patch(row_bg)

            if i > 0:
                ax.plot(
                    [x + 0.15, x + box_w - 0.15],
                    [ry + row_h, ry + row_h],
                    color="#e0e0e0",
                    lw=0.6,
                )

            ty = ry + row_h / 2
            ax.text(
                x + 0.3,
                ty,
                name,
                ha="left",
                va="center",
                fontsize=8.5,
                fontweight="bold" if is_leader else "normal",
                color=dark if is_leader else "#555555",
                fontfamily="sans-serif",
            )
            if prob > 0.0:
                ax.text(
                    x + box_w - 0.25,
                    ty,
                    f"{prob:.1%}",
                    ha="right",
                    va="center",
                    fontsize=8,
                    color=accent if is_leader else "#888888",
                    fontweight="bold" if is_leader else "normal",
                    fontfamily="sans-serif",
                )

        # Left accent stripe
        stripe = mpatches.FancyBboxPatch(
            (x, y),
            0.12,
            box_h,
            boxstyle=mpatches.BoxStyle.Round(pad=0.02),
            facecolor=accent,
            edgecolor="none",
        )
        ax.add_patch(stripe)

    @staticmethod
    def _draw_connector(
        ax: matplotlib.axes.Axes,
        parent_xy: tuple[float, float],
        child1_xy: tuple[float, float],
        child2_xy: tuple[float, float],
    ) -> None:
        """Draw bracket connector lines from two child slots to a parent slot.

        The connector uses three segments: two horizontal stubs from the child
        right-edges to a shared vertical mid-point, then one horizontal segment
        from that mid-point to the parent left-edge.

        Args:
            ax: Target matplotlib axes.
            parent_xy: (x, y) of the parent slot's connection point (left edge
                centre).
            child1_xy: (x, y) of the first child slot's right edge centre.
            child2_xy: (x, y) of the second child slot's right edge centre.
        """
        mid_x = (parent_xy[0] + child1_xy[0]) / 2
        color = "#cccccc"
        lw = 1.0

        ax.plot([child1_xy[0], mid_x], [child1_xy[1], child1_xy[1]], color=color, lw=lw)
        ax.plot([child2_xy[0], mid_x], [child2_xy[1], child2_xy[1]], color=color, lw=lw)
        ax.plot([mid_x, mid_x], [child1_xy[1], child2_xy[1]], color=color, lw=lw)
        ax.plot([mid_x, parent_xy[0]], [parent_xy[1], parent_xy[1]], color=color, lw=lw)
