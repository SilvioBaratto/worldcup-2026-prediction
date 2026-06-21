"""Clean and preprocess raw match data into analysis-ready training data.

Output schema (train_data.csv):
    HOME_TEAM, AWAY_TEAM,
    GOALS_home, SHOTS_home, SHOTS_ON_TARGET_home, POSSESSION_home, PASS_PCT_home,
    GOALS_away, SHOTS_away, SHOTS_ON_TARGET_away, POSSESSION_away, PASS_PCT_away,
    HOME_WIN
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from worldcup_playoff.config import DataConfig, FeaturesConfig
from worldcup_playoff.data.loader import validate_matches_df

logger = logging.getLogger(__name__)

# Columns produced in train_data.csv (mirrors SHARED CONTRACT)
_OUTPUT_COLUMNS = [
    "HOME_TEAM",
    "AWAY_TEAM",
    "GOALS_home",
    "SHOTS_home",
    "SHOTS_ON_TARGET_home",
    "POSSESSION_home",
    "PASS_PCT_home",
    "GOALS_away",
    "SHOTS_away",
    "SHOTS_ON_TARGET_away",
    "POSSESSION_away",
    "PASS_PCT_away",
    "HOME_WIN",
]

# Columns expected in match_details.csv
_DETAIL_COLUMNS = [
    "MATCH_ID",
    "GOALS_home",
    "SHOTS_home",
    "SHOTS_ON_TARGET_home",
    "POSSESSION_home",
    "PASS_PCT_home",
    "GOALS_away",
    "SHOTS_away",
    "SHOTS_ON_TARGET_away",
    "POSSESSION_away",
    "PASS_PCT_away",
]


class DataCleaner:
    """Transforms raw match data into analysis-ready training data.

    Pipeline steps:

    1. Validate input DataFrames.
    2. Merge ``matches_df`` with ``details_df`` (when provided) to attach
       the 10 feature columns per row.
    3. Filter rows by ``min_date``.
    4. Drop drawn matches (``HOME_GOALS == AWAY_GOALS``).
    5. Apply epsilon to zero possession/pass-pct values.
    6. Drop rows with NaN in any feature column.
    7. Apply train cutoff date.
    8. Add ``HOME_WIN`` binary target.
    9. Select and order output columns to match train_data.csv schema.
    """

    def __init__(self, config: DataConfig, features_config: FeaturesConfig) -> None:
        self._config = config
        self._features = features_config

    def clean(
        self,
        matches_df: pd.DataFrame,
        details_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Full cleaning pipeline.

        Args:
            matches_df: Raw matches DataFrame (from matches.csv).
            details_df: Optional match details DataFrame (from match_details.csv).
                When absent, feature columns are taken directly from matches_df
                if present, otherwise the cleaner raises an error.

        Returns:
            Cleaned DataFrame with exactly the ``_OUTPUT_COLUMNS`` schema.

        Raises:
            ValueError: If required feature columns cannot be assembled.
        """
        validate_matches_df(matches_df)

        df = self._sort_by_date(matches_df)
        df = self._filter_by_min_date(df)
        df = self._drop_draws(df)
        df = self._attach_features(df, details_df)
        df = self._fix_zero_percentages(df)
        df = self._drop_nan_features(df)
        df = self._apply_train_cutoff(df)
        df = self._add_home_win(df)
        df = self._select_output_columns(df)

        logger.info("Cleaning complete: %d rows retained", len(df))
        return df

    # ------------------------------------------------------------------
    # Pipeline steps — each is a pure transformation on the DataFrame
    # ------------------------------------------------------------------

    def _sort_by_date(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.sort_values(by="DATE").reset_index(drop=True)

    def _filter_by_min_date(self, df: pd.DataFrame) -> pd.DataFrame:
        mask = df["DATE"] >= self._config.min_date
        filtered = df.loc[mask].reset_index(drop=True)
        logger.debug(
            "Filtered to matches after %s: %d -> %d rows",
            self._config.min_date,
            len(df),
            len(filtered),
        )
        return filtered

    def _drop_draws(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove drawn matches where HOME_GOALS == AWAY_GOALS.

        A knockout tie always produces a winner (extra time + penalties are
        collapsed into a single win/loss), so draws in the historical dataset
        are regular-season group matches — irrelevant for predicting elimination
        outcomes and confusing for a binary classifier.
        """
        before = len(df)
        mask = df["HOME_GOALS"] != df["AWAY_GOALS"]
        filtered = df.loc[mask].reset_index(drop=True)
        logger.debug("Dropped %d drawn matches", before - len(filtered))
        return filtered

    def _attach_features(
        self, df: pd.DataFrame, details_df: pd.DataFrame | None
    ) -> pd.DataFrame:
        """Merge match_details statistics onto the main DataFrame.

        If ``details_df`` is provided it is joined on ``MATCH_ID``.  Rows
        without a matching detail record receive NaN feature values (these
        are dropped later by ``_drop_nan_features``).

        If ``details_df`` is ``None`` the method checks whether the 10 feature
        columns are already present (e.g. when the caller pre-merged them)
        and raises ``ValueError`` otherwise.
        """
        if details_df is not None:
            # Keep only the columns we need from details
            detail_cols = [c for c in _DETAIL_COLUMNS if c in details_df.columns]
            merged = df.merge(
                details_df[detail_cols],
                on="MATCH_ID",
                how="left",
                suffixes=("", "_detail"),
            )
            # For GOALS the matches.csv already has HOME_GOALS / AWAY_GOALS;
            # the detail file duplicates them — prefer the detail version.
            if "GOALS_home" in merged.columns and "HOME_GOALS" in merged.columns:
                merged["GOALS_home"] = merged["GOALS_home"].fillna(
                    merged["HOME_GOALS"]
                )
            else:
                merged["GOALS_home"] = merged["HOME_GOALS"]

            if "GOALS_away" in merged.columns and "AWAY_GOALS" in merged.columns:
                merged["GOALS_away"] = merged["GOALS_away"].fillna(
                    merged["AWAY_GOALS"]
                )
            else:
                merged["GOALS_away"] = merged["AWAY_GOALS"]

            return merged

        # No details_df: ensure feature columns exist on the matches DataFrame
        missing_features = [c for c in self._features.selected if c not in df.columns]
        if missing_features:
            raise ValueError(
                f"Feature columns {missing_features} not found and no details_df provided. "
                "Run 'worldcup download --only match-details' or supply a details DataFrame."
            )
        # Copy GOALS from goals columns if not already named
        if "GOALS_home" not in df.columns:
            df = df.copy()
            df["GOALS_home"] = df["HOME_GOALS"]
            df["GOALS_away"] = df["AWAY_GOALS"]
        return df

    def _fix_zero_percentages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace zero values in percentage columns with epsilon.

        Covers ``POSSESSION_home``, ``POSSESSION_away``, ``PASS_PCT_home``,
        ``PASS_PCT_away`` — any selected feature whose name contains 'PCT'
        or 'POSSESSION'.  After patching zeros, rows that still contain NaN
        in any feature column are dropped so classifiers receive a clean matrix.
        """
        eps = self._config.epsilon
        df = df.copy()
        pct_cols = [
            c
            for c in self._features.selected
            if "PCT" in c or "POSSESSION" in c
        ]
        for col in pct_cols:
            if col not in df.columns:
                continue
            mask = df[col] == 0
            if mask.any():
                df.loc[mask, col] = eps
                logger.debug("Fixed %d zero values in %s", int(mask.sum()), col)
        return df

    def _drop_nan_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows with NaN in any feature column."""
        available = [c for c in self._features.selected if c in df.columns]
        before = len(df)
        df = df.dropna(subset=available).reset_index(drop=True)
        dropped = before - len(df)
        if dropped:
            logger.debug("Dropped %d rows with NaN in feature columns", dropped)
        return df

    def _apply_train_cutoff(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep only matches before the training cutoff date."""
        mask = df["DATE"] < self._config.train_cutoff_date
        filtered = df.loc[mask].reset_index(drop=True)
        logger.debug(
            "Applied cutoff %s: %d -> %d rows",
            self._config.train_cutoff_date,
            len(df),
            len(filtered),
        )
        return filtered

    def _add_home_win(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add binary HOME_WIN column: 1 if HOME_GOALS > AWAY_GOALS else 0."""
        df = df.copy()
        df["HOME_WIN"] = (df["HOME_GOALS"] > df["AWAY_GOALS"]).astype(int)
        return df

    def _select_output_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename team ID columns to HOME_TEAM / AWAY_TEAM and select output schema.

        The matches.csv stores team names directly in HOME_TEAM / AWAY_TEAM
        so no ID-to-name mapping is required (unlike the NBA original which
        uses numeric TEAM_IDs mapped through teams.csv).
        """
        available = [c for c in _OUTPUT_COLUMNS if c in df.columns]
        missing = set(_OUTPUT_COLUMNS) - set(available)
        if missing:
            raise ValueError(
                f"Cannot produce train_data.csv — missing columns: {missing}. "
                "Ensure match_details.csv was built and merged correctly."
            )
        return df[_OUTPUT_COLUMNS].reset_index(drop=True)

    def write(
        self,
        matches_df: pd.DataFrame,
        details_df: pd.DataFrame | None = None,
        root: Path | None = None,
    ) -> pd.DataFrame:
        """Clean the data and write train_data.csv to ``DataConfig.output_path``.

        Args:
            matches_df: Raw matches DataFrame.
            details_df: Optional match details DataFrame.
            root: Root directory prepended to ``output_path``. Defaults to cwd.

        Returns:
            The cleaned DataFrame.
        """
        cleaned = self.clean(matches_df, details_df)
        root = root or Path.cwd()
        output_path = root / self._config.output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned.to_csv(output_path, index=False)
        logger.info("Wrote train_data to %s (%d rows)", output_path, len(cleaned))
        return cleaned
