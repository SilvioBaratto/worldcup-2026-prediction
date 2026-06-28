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
from worldcup_playoff.visualization.forecast_plots import (  # noqa: E402
    plot_round_advancement,
    plot_title_odds,
)

__all__ = ["ResultPlotter", "plot_title_odds", "plot_round_advancement"]

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
        slot_teams: dict[int, list[tuple[str, str]]] | None = None,
        slot_scores: dict[str, tuple[int, int]] | None = None,
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
        slots_by_round = slot_teams or _build_round_slots(bracket.matchups)
        if not slots_by_round:
            logger.warning("plot_bracket called with empty matchup list — nothing to draw.")
            return

        max_round = max(slots_by_round)
        total_rounds = max_round + 1

        # --- Layout constants (tweaked for left-to-right single side) ---
        BOX_W: float = 6.2
        BOX_H: float = 1.7
        ROW_GAP: float = 0.75
        COL_SPACING: float = 7.9
        MARGIN_X: float = 1.0
        MARGIN_TOP: float = 4.8  # room for title + reading guide + column labels
        MARGIN_BOT: float = 4.2  # room for the legend below the bracket

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

        # --- Reading guide (one line; full key is in the legend below) ---
        ax.text(
            fig_w / 2,
            fig_h - 1.45,
            "Predicted knockout path — see the legend at the bottom for how to read the numbers.",
            ha="center",
            va="top",
            fontsize=8,
            style="italic",
            color="#6b7280",
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
        label_y = fig_h - 2.7
        for rnd in range(total_rounds):
            if positions[rnd]:
                col_cx = positions[rnd][0][0] + BOX_W / 2
                label = _round_label(rnd, total_rounds)
                if rnd == max_round:
                    label = f"{label}  ·  champion odds"
                ax.text(
                    col_cx,
                    label_y,
                    label,
                    ha="center",
                    va="top",
                    fontsize=8,
                    fontweight="bold" if rnd == max_round else "normal",
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
                score = (slot_scores or {}).get(f"{home}|{away}")
                goals = {home: score[0], away: score[1]} if score else None
                self._draw_team_box(ax, bx, by, team_probs, palette, BOX_W, BOX_H, goals)

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

        # --- Legend (key for probabilities + predicted score) ---
        legend_w = min(fig_w * 0.84, 40.0)
        legend_h = 3.4
        self._draw_legend(ax, (fig_w - legend_w) / 2, 0.6, legend_w, legend_h)

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
        goals: dict[str, int] | None = None,
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
            name_fs = 8.5 if len(name) <= 16 else 7.5  # shrink long names to avoid overlap
            ax.text(
                x + 0.45,
                ty,
                name,
                ha="left",
                va="center",
                fontsize=name_fs,
                fontweight="bold" if is_leader else "normal",
                color=dark if is_leader else "#555555",
                fontfamily="sans-serif",
            )
            # Most-likely goals — read the column vertically as the scoreline.
            if goals is not None and name in goals:
                ax.text(
                    x + box_w - 2.05,
                    ty,
                    str(goals[name]),
                    ha="center",
                    va="center",
                    fontsize=10.5,
                    color=dark if is_leader else "#777777",
                    fontweight="bold",
                    fontfamily="sans-serif",
                )
            if prob > 0.0:
                ax.text(
                    x + box_w - 0.35,
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

    def _draw_legend(
        self,
        ax: matplotlib.axes.Axes,
        x0: float,
        y0: float,
        w: float,
        h: float,
    ) -> None:
        """Draw the bracket key: what the percentage, scoreline, and colours mean."""
        panel = mpatches.FancyBboxPatch(
            (x0, y0), w, h,
            boxstyle=mpatches.BoxStyle.Round(pad=0.12),
            facecolor="white", edgecolor="#cccccc", linewidth=1.0,
        )
        ax.add_patch(panel)
        ax.text(
            x0 + 0.5, y0 + h - 0.45, "HOW TO READ",
            fontsize=9, fontweight="bold", color="#0d1b2a",
            ha="left", va="top", fontfamily="sans-serif",
        )

        seg = w / 4.0
        text_dx = 1.9
        row_y = y0 + h * 0.52

        def _label(i: int, l1: str, l2: str) -> float:
            tx = x0 + i * seg + text_dx
            ax.text(tx, row_y + 0.28, l1, fontsize=8, fontweight="bold",
                    color="#333333", ha="left", va="center", fontfamily="sans-serif")
            ax.text(tx, row_y - 0.30, l2, fontsize=7.5, color="#666666",
                    ha="left", va="center", fontfamily="sans-serif")
            return x0 + i * seg + 0.55

        # 1 — winner swatch (shaded row + accent stripe)
        ex = _label(0, "Shaded bold row", "= predicted winner")
        sw = mpatches.FancyBboxPatch(
            (ex, row_y - 0.32), 1.1, 0.64, boxstyle=mpatches.BoxStyle.Round(pad=0.03),
            facecolor="#d6eaf8", edgecolor="none")
        ax.add_patch(sw)
        ax.add_patch(mpatches.FancyBboxPatch(
            (ex, row_y - 0.32), 0.12, 0.64, boxstyle=mpatches.BoxStyle.Round(pad=0.01),
            facecolor="#2e86c1", edgecolor="none"))

        # 2 — percentage
        ex = _label(1, "63%  = chance to win", "at this stage")
        ax.text(ex + 0.55, row_y, "63%", fontsize=11, fontweight="bold",
                color="#2e86c1", ha="center", va="center", fontfamily="sans-serif")

        # 3 — most-likely scoreline
        ex = _label(2, "1 – 0  = most likely", "score (goals per team)")
        ax.text(ex + 0.5, row_y, "1–0", fontsize=12, fontweight="bold",
                color="#0d1b2a", ha="center", va="center", fontfamily="sans-serif")

        # 4 — champion
        ex = _label(3, "Gold = CHAMPION", "(title probability)")
        ax.add_patch(mpatches.FancyBboxPatch(
            (ex, row_y - 0.32), 1.1, 0.64, boxstyle=mpatches.BoxStyle.Round(pad=0.03),
            facecolor="#fef9e7", edgecolor="#d4af37", linewidth=1.5))

        ax.text(
            x0 + w / 2, y0 + 0.42,
            "Round of 32 = head-to-head (the two %s sum to 100%).  Later rounds show only the "
            "2 likeliest of all teams that can reach each slot — and the Final % is the title chance.",
            fontsize=7.5, style="italic", color="#6b7280",
            ha="center", va="center", fontfamily="sans-serif",
        )

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
