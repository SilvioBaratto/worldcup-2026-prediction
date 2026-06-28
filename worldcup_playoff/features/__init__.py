"""Football-only covariate assembly for the World Cup 2026 prediction model.

Exports the public symbols needed to build per-match feature vectors:
- Confederation lookup: ``CONFEDERATIONS``, ``CONFEDERATION_MAP``, ``confederation_of``,
  ``resolve_ranking``, ``RankingResolution``.
- Time-aware helpers: ``sort_chronological``, ``recent_form``, ``rest_days``, ``goal_difference``.
- Feature assembler: ``MatchFeatures``, ``FeatureBuilder``, ``build_features``, ``to_frame``,
  ``FEATURE_COLUMNS``.
"""

from __future__ import annotations

from worldcup_playoff.features.build import (
    FEATURE_COLUMNS,
    FeatureBuilder,
    MatchFeatures,
    TeamAbilities,
    build_features,
    to_frame,
)
from worldcup_playoff.features.confederation import (
    CONFEDERATION_MAP,
    CONFEDERATIONS,
    RankingResolution,
    confederation_of,
    resolve_ranking,
)
from worldcup_playoff.features.timeaware import (
    goal_difference,
    recent_form,
    rest_days,
    sort_chronological,
)

__all__ = [
    "CONFEDERATIONS",
    "CONFEDERATION_MAP",
    "FEATURE_COLUMNS",
    "FeatureBuilder",
    "MatchFeatures",
    "RankingResolution",
    "TeamAbilities",
    "build_features",
    "confederation_of",
    "goal_difference",
    "recent_form",
    "resolve_ranking",
    "rest_days",
    "sort_chronological",
    "to_frame",
]
