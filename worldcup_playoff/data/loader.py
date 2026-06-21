"""Load raw CSV datasets into validated DataFrames."""

from __future__ import annotations

import difflib
import logging
from pathlib import Path

import pandas as pd

from worldcup_playoff.config import DataConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Required column schemas (column name -> expected pandas dtype string)
# ---------------------------------------------------------------------------

REQUIRED_MATCHES_COLUMNS: dict[str, str] = {
    "MATCH_ID": "int64",
    "DATE": "object",
    "HOME_TEAM": "object",
    "AWAY_TEAM": "object",
    "HOME_GOALS": "int64",
    "AWAY_GOALS": "int64",
    "COMPETITION": "object",
    "SEASON": "int64",
}

REQUIRED_TEAMS_COLUMNS: dict[str, str] = {
    "TEAM_ID": "int64",
    "NAME": "object",
}


def _validate_dataframe(
    df: pd.DataFrame,
    required: dict[str, str],
    label: str,
) -> None:
    """Check column presence and dtypes, raising ``ValueError`` on mismatch.

    Args:
        df: DataFrame to validate.
        required: Mapping of column name to expected dtype string.
        label: Human-readable label used in error messages (e.g. ``"matches"``).

    Raises:
        ValueError: If any required column is absent or has an incompatible dtype.
    """
    missing = set(required) - set(df.columns)
    if missing:
        parts: list[str] = []
        for col in sorted(missing):
            matches = difflib.get_close_matches(col, df.columns.tolist(), n=1, cutoff=0.7)
            if matches:
                parts.append(f"  '{col}' — did you mean '{matches[0]}'?")
            else:
                parts.append(f"  '{col}'")
        detail = "\n".join(parts)
        raise ValueError(f"{label} DataFrame missing required column(s):\n{detail}")

    _NUMERIC_DTYPES = {"int64", "float64", "int32", "float32"}
    # pandas 2.0+ on Python 3.14 uses the 'str' (ArrowDtype string) dtype
    # for string columns; treat 'str' and 'object' as interchangeable.
    _STRING_DTYPES = {"object", "str", "string"}

    type_errors: list[str] = []
    for col, expected in required.items():
        actual = str(df[col].dtype)
        if actual != expected:
            # Allow numeric type compatibility (int64 <-> float64)
            if actual in _NUMERIC_DTYPES and expected in _NUMERIC_DTYPES:
                continue
            # Allow string type compatibility (object <-> str <-> string)
            if actual in _STRING_DTYPES and expected in _STRING_DTYPES:
                continue
            type_errors.append(f"  '{col}' has dtype '{actual}', expected '{expected}'")

    if type_errors:
        detail = "\n".join(type_errors)
        raise ValueError(f"{label} DataFrame has wrong column dtype(s):\n{detail}")


def validate_matches_df(df: pd.DataFrame) -> None:
    """Validate that a matches DataFrame has the required schema."""
    _validate_dataframe(df, REQUIRED_MATCHES_COLUMNS, "matches")


def validate_teams_df(df: pd.DataFrame) -> None:
    """Validate that a teams DataFrame has the required schema."""
    _validate_dataframe(df, REQUIRED_TEAMS_COLUMNS, "teams")


class DataLoader:
    """Reads raw CSV files into DataFrames with column validation."""

    def __init__(self, config: DataConfig, root: Path | None = None) -> None:
        self._config = config
        self._root = root or Path.cwd()

    def load_matches(self) -> pd.DataFrame:
        """Load matches.csv into a validated DataFrame.

        Returns:
            DataFrame conforming to ``REQUIRED_MATCHES_COLUMNS``.

        Raises:
            ValueError: If ``matches_path`` is not configured.
            ValueError: If the loaded DataFrame fails schema validation.
        """
        if self._config.matches_path is None:
            raise ValueError(
                "matches_path is not configured; run 'worldcup download' first "
                "or set data.matches_path in your config"
            )
        path = self._root / self._config.matches_path
        logger.info("Loading matches from %s", path)
        df = pd.read_csv(path)
        validate_matches_df(df)
        return df

    def load_teams(self) -> pd.DataFrame:
        """Load teams.csv into a validated DataFrame.

        Returns:
            DataFrame conforming to ``REQUIRED_TEAMS_COLUMNS``.

        Raises:
            ValueError: If ``teams_path`` is not configured.
            ValueError: If the loaded DataFrame fails schema validation.
        """
        if self._config.teams_path is None:
            raise ValueError(
                "teams_path is not configured; run 'worldcup download' first "
                "or set data.teams_path in your config"
            )
        path = self._root / self._config.teams_path
        logger.info("Loading teams from %s", path)
        df = pd.read_csv(path)
        validate_teams_df(df)
        return df
