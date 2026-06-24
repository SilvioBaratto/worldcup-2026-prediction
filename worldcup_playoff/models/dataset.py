"""Time-aware match dataset utilities for Cycle-4 models.

Provides a single source of truth for target derivation and chronological
train/test splitting so both the hybrid and ordered-logit models share
identical data conventions.

Pure pandas/numpy only — no sklearn/statsmodels — so this module loads fast
in any context, including test collection.

The feature frame produced by ``build_features`` uses lowercase column names
(``date``, ``home_goals``, ``away_goals``) — distinct from the martj42 raw
schema (``DATE``, ``HOME_GOALS``, ``AWAY_GOALS``). Sorting is applied to the
``date`` column directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

__all__ = [
    "MatchDataset",
    "add_targets",
    "build_dataset",
    "chronological_split",
    "outcome_label",
    "played_only",
]


# ---------------------------------------------------------------------------
# Value object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatchDataset:
    """Immutable container produced by ``build_dataset``.

    Attributes
    ----------
    train:        Played matches in the chronological training slice.
    test:         Played matches in the chronological test slice.
    feature_cols: Column names to use as model inputs.  When omitted (empty
                  list), the consuming model auto-derives the column set from
                  the training frame by excluding the standard forbidden
                  columns (targets / identity / metadata).
    """

    train: pd.DataFrame
    test: pd.DataFrame
    feature_cols: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def outcome_label(home_goals: int, away_goals: int) -> int:
    """Return ordered encoding: away-win=0, draw=1, home-win=2."""
    if home_goals > away_goals:
        return 2
    if home_goals == away_goals:
        return 1
    return 0


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with ``y_outcome`` (Int64) and ``y_margin`` added.

    ``y_outcome`` uses the away<draw<home encoding from ``outcome_label``.
    ``y_margin`` = home_goals − away_goals (signed; used for group tiebreaks).
    Rows with NA goals receive NA targets.
    """
    out = df.copy()
    out["y_outcome"] = (
        out.apply(
            lambda r: outcome_label(int(r["home_goals"]), int(r["away_goals"]))
            if pd.notna(r["home_goals"]) and pd.notna(r["away_goals"])
            else pd.NA,
            axis=1,
        )
        .astype("Int64")
    )
    out["y_margin"] = (out["home_goals"] - out["away_goals"]).astype("Int64")
    return out


def played_only(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows where both ``home_goals`` and ``away_goals`` are non-NA."""
    mask = df["home_goals"].notna() & df["away_goals"].notna()
    return df[mask].reset_index(drop=True)


def _sort_by_date(df: pd.DataFrame) -> pd.DataFrame:
    """Sort feature-frame by ``date`` ascending, NaT last. Stable, no RNG."""
    return (
        df.assign(_d=pd.to_datetime(df["date"], errors="coerce"))
        .sort_values("_d", kind="stable", na_position="last")
        .drop(columns=["_d"])
        .reset_index(drop=True)
    )


def chronological_split(
    df: pd.DataFrame, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split *df* positionally: last ``floor(n * test_size)`` rows → test.

    No shuffling, no RNG — purely positional so the result is deterministic.
    Re-asserts chronological order by ``date`` before splitting (self-defensive).

    Returns
    -------
    (train, test) DataFrames with reset indices.
    """
    ordered = _sort_by_date(df)
    n = len(ordered)
    n_test = math.floor(n * test_size)
    n_train = n - n_test
    train = ordered.iloc[:n_train].reset_index(drop=True)
    test = ordered.iloc[n_train:].reset_index(drop=True)
    return train, test


def build_dataset(
    features: pd.DataFrame,
    test_size: float,
    feature_cols: list[str],
) -> MatchDataset:
    """Apply ``played_only`` then ``chronological_split`` and return a frozen dataset.

    ``played_only`` is applied FIRST so unplayed WC2026 fixtures (NA goals,
    sorted last by date) never contaminate the test slice.
    """
    played = played_only(features)
    train, test = chronological_split(played, test_size=test_size)
    return MatchDataset(train=train, test=test, feature_cols=list(feature_cols))
