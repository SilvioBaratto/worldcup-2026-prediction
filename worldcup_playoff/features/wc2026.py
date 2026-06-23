"""WC2026 current-state feature frame.

Produces feature rows for **unplayed** WC2026 fixtures by running the full
issue-#9 ``FeatureBuilder`` over the complete chronological history and then
filtering to the rows that correspond to unplayed ``FIFA World Cup`` matches.

Running the builder over the full history (not over the WC2026 subset alone)
is essential: pre-match form, rest days, and Elo ratings must reflect ALL of a
team's prior football — not just their prior WC matches — to avoid leakage and
to produce the same covariate values the training pipeline would produce.

The positional mask technique (``sort_chronological(df)['TOURNAMENT'] ==
'FIFA World Cup'``) is row-for-row aligned with the ``FeatureBuilder`` output
since both internally call ``sort_chronological`` with the same key.

A ``tournament`` metadata column is appended to the output (always
``'FIFA World Cup'``) so callers can identify the WC2026 rows without
re-joining on ``(date, home_team, away_team)`` — a join that neutral-site
duplicates and same-day fixtures can corrupt.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from worldcup_playoff.config import FeatureBuildConfig
from worldcup_playoff.data.live import live_fixtures_to_df  # re-exported as part of wc2026 surface
from worldcup_playoff.features.build import FeatureBuilder
from worldcup_playoff.features.timeaware import sort_chronological
from worldcup_playoff.simulation.poisson import TeamAbilities

if TYPE_CHECKING:
    from worldcup_playoff.data.elo import EloResult

logger = logging.getLogger(__name__)

_WC_TOURNAMENT = "FIFA World Cup"

__all__ = ["live_fixtures_to_df", "wc2026_features"]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _unplayed_wc_mask(sorted_df: pd.DataFrame) -> pd.Series:
    """Positional boolean mask: rows that are unplayed FIFA World Cup fixtures."""
    return (sorted_df["TOURNAMENT"] == _WC_TOURNAMENT) & sorted_df["HOME_GOALS"].isna()


def _filter_wc_rows(features_df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """Filter feature rows by positional mask and tag with tournament metadata."""
    result = features_df[mask.values].copy()
    result["tournament"] = _WC_TOURNAMENT
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _to_elo_df(elo_result: EloResult) -> pd.DataFrame:
    """Extract home_elo/away_elo per match from an EloResult (aligned to sorted df)."""
    return pd.DataFrame(
        [{"home_elo": d.home_elo, "away_elo": d.away_elo} for d in elo_result.match_diffs]
    )


def wc2026_features(
    results_df: pd.DataFrame,
    elo: EloResult | pd.DataFrame,
    abilities: TeamAbilities,
    *,
    config: FeatureBuildConfig | None = None,
    ranking: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Return feature rows for every unplayed WC2026 fixture in *results_df*.

    Parameters
    ----------
    results_df: Full martj42 results frame (historical + WC2026 fixtures).
                Must be passable to ``FeatureBuilder.build``; includes both
                played and unplayed rows.
    elo:        Either an ``EloResult`` from ``compute_elo`` or a pre-built elo DataFrame
                with ``home_elo`` / ``away_elo`` columns aligned to
                ``sort_chronological(results_df)``.
    abilities:  Dixon-Coles attack/defence abilities.
    config:     Feature-build settings; uses ``FeatureBuildConfig()`` defaults
                when ``None``.
    ranking:    Optional ``{team: ranking_points}`` dict; falls back to the
                static confederation map when ``None``.

    Returns
    -------
    DataFrame with one row per unplayed WC2026 fixture, the full feature schema
    from ``FeatureBuilder`` plus a ``tournament`` metadata column, and
    ``home_goals``/``away_goals`` as ``<NA>``.
    """
    from worldcup_playoff.data.elo import EloResult as _EloResult  # avoid circular at import time

    cfg = config or FeatureBuildConfig()
    sorted_df = sort_chronological(results_df)
    elo_df = _to_elo_df(elo) if isinstance(elo, _EloResult) else elo
    all_features = FeatureBuilder(cfg, ranking=ranking).build(sorted_df, elo_df, abilities)
    mask = _unplayed_wc_mask(sorted_df)
    return _filter_wc_rows(all_features, mask)
