"""No-key CC0 loader for martj42/international_results datasets."""

from __future__ import annotations

import difflib
import io
import logging
from pathlib import Path

import pandas as pd
import requests

from worldcup_playoff.config import Martj42Config
from worldcup_playoff.data.crosswalk import normalize_series

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_RESULTS_RENAME: dict[str, str] = {
    "date": "DATE",
    "home_team": "HOME_TEAM",
    "away_team": "AWAY_TEAM",
    "home_score": "HOME_GOALS",
    "away_score": "AWAY_GOALS",
    "tournament": "TOURNAMENT",
    "neutral": "NEUTRAL",
}

_SHOOTOUTS_RENAME: dict[str, str] = {
    "date": "DATE",
    "home_team": "HOME_TEAM",
    "away_team": "AWAY_TEAM",
    "winner": "WINNER",
    "first_shooter": "FIRST_SHOOTER",
}

_GOALSCORERS_RENAME: dict[str, str] = {
    "date": "DATE",
    "home_team": "HOME_TEAM",
    "away_team": "AWAY_TEAM",
    "team": "TEAM",
    "scorer": "SCORER",
    "minute": "MINUTE",
    "own_goal": "OWN_GOAL",
    "penalty": "PENALTY",
}

# ---------------------------------------------------------------------------
# Required column schemas (column name -> expected dtype string)
# ---------------------------------------------------------------------------

REQUIRED_MARTJ42_RESULTS_COLUMNS: dict[str, str] = {
    "DATE": "object",
    "HOME_TEAM": "object",
    "AWAY_TEAM": "object",
    "HOME_GOALS": "Int64",
    "AWAY_GOALS": "Int64",
    "TOURNAMENT": "object",
    "NEUTRAL": "bool",
}

REQUIRED_MARTJ42_SHOOTOUTS_COLUMNS: dict[str, str] = {
    "DATE": "object",
    "HOME_TEAM": "object",
    "AWAY_TEAM": "object",
    "WINNER": "object",
    "FIRST_SHOOTER": "object",
}

REQUIRED_MARTJ42_GOALSCORERS_COLUMNS: dict[str, str] = {
    "DATE": "object",
    "HOME_TEAM": "object",
    "AWAY_TEAM": "object",
    "TEAM": "object",
    "SCORER": "object",
    "MINUTE": "Int64",
    "OWN_GOAL": "bool",
    "PENALTY": "bool",
}

# ---------------------------------------------------------------------------
# Boolean column parser (handles both string and auto-inferred bool input)
# ---------------------------------------------------------------------------


def _parse_bool_column(series: pd.Series) -> pd.Series:
    """Coerce TRUE/FALSE (string or auto-detected bool) to numpy bool.

    pandas may auto-detect 'TRUE'/'FALSE' CSV strings as bool; converting to
    str first normalises both paths: str(True)='True' -> upper='TRUE' -> True.
    """
    return series.astype(str).str.upper().map({"TRUE": True, "FALSE": False}).astype(bool)


# dtype family sets used by the validator
_INT64_DTYPES = {"Int64"}
_BOOL_DTYPES = {"bool", "boolean"}
_NUMERIC_DTYPES = {"int64", "float64", "int32", "float32"}
_STRING_DTYPES = {"object", "str", "string"}

# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


def _validate_martj42(df: pd.DataFrame, required: dict[str, str], label: str) -> None:
    """Raise ValueError for missing columns or incompatible dtypes.

    Extends loader.py's family logic to accept nullable Int64 and bool.
    """
    missing = set(required) - set(df.columns)
    if missing:
        parts = []
        for col in sorted(missing):
            near = difflib.get_close_matches(col, df.columns.tolist(), n=1, cutoff=0.7)
            hint = f" — did you mean '{near[0]}'?" if near else ""
            parts.append(f"  '{col}'{hint}")
        raise ValueError(f"{label} DataFrame missing required column(s):\n" + "\n".join(parts))

    type_errors: list[str] = []
    for col, expected in required.items():
        actual = str(df[col].dtype)
        if actual == expected:
            continue
        if expected in _INT64_DTYPES:
            type_errors.append(f"  '{col}' has dtype '{actual}', expected '{expected}'")
            continue
        if expected in _BOOL_DTYPES:
            if df[col].dtype.kind != "b":
                type_errors.append(f"  '{col}' has dtype '{actual}', expected bool-like")
            continue
        if actual in _NUMERIC_DTYPES and expected in _NUMERIC_DTYPES:
            continue
        if actual in _STRING_DTYPES and expected in _STRING_DTYPES:
            continue
        type_errors.append(f"  '{col}' has dtype '{actual}', expected '{expected}'")

    if type_errors:
        raise ValueError(f"{label} DataFrame has wrong column dtype(s):\n" + "\n".join(type_errors))


def validate_results_df(df: pd.DataFrame) -> None:
    """Validate that a results DataFrame conforms to the martj42 results schema."""
    _validate_martj42(df, REQUIRED_MARTJ42_RESULTS_COLUMNS, "martj42_results")


def validate_shootouts_df(df: pd.DataFrame) -> None:
    """Validate that a shootouts DataFrame conforms to the martj42 shootouts schema."""
    _validate_martj42(df, REQUIRED_MARTJ42_SHOOTOUTS_COLUMNS, "martj42_shootouts")


def validate_goalscorers_df(df: pd.DataFrame) -> None:
    """Validate that a goalscorers DataFrame conforms to the martj42 goalscorers schema."""
    _validate_martj42(df, REQUIRED_MARTJ42_GOALSCORERS_COLUMNS, "martj42_goalscorers")


# ---------------------------------------------------------------------------
# Convenience filter
# ---------------------------------------------------------------------------


def wc2026_schedule(df: pd.DataFrame) -> pd.DataFrame:
    """Return 2026 FIFA World Cup rows (played + unplayed), excluding all prior editions."""
    is_wc = df["TOURNAMENT"] == "FIFA World Cup"
    is_2026 = pd.to_datetime(df["DATE"], errors="coerce").dt.year == 2026
    return df[is_wc & is_2026].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class Martj42Loader:
    """Cache-first loader for the martj42/international_results CC0 datasets."""

    def __init__(
        self,
        config: Martj42Config,
        cache_dir: Path | None = None,
        base_url: str | None = None,
    ) -> None:
        self._config = config
        self._cache_dir = cache_dir or config.cache_dir
        self._base_url = base_url or config.base_url

    # --- public API ---

    def load_results(self) -> pd.DataFrame:
        """Return the results DataFrame coerced to the internal schema."""
        return self._coerce_results(self._fetch_raw(self._config.results_file))

    def load_shootouts(self) -> pd.DataFrame:
        """Return the shootouts DataFrame coerced to the internal schema."""
        return self._coerce_shootouts(self._fetch_raw(self._config.shootouts_file))

    def load_goalscorers(self) -> pd.DataFrame:
        """Return the goalscorers DataFrame coerced to the internal schema."""
        return self._coerce_goalscorers(self._fetch_raw(self._config.goalscorers_file))

    def wc2026_schedule(self) -> pd.DataFrame:
        """Return FIFA World Cup rows (played + unplayed) from results."""
        return wc2026_schedule(self.load_results())

    # --- private helpers ---

    def _fetch_raw(self, filename: str) -> pd.DataFrame:
        path = self._cache_dir / filename
        if path.exists():
            logger.info("Loading %s from cache: %s", filename, path)
            return pd.read_csv(path)
        return self._download_and_cache(filename, path)

    def _download_and_cache(self, filename: str, path: Path) -> pd.DataFrame:
        url = self._base_url + filename
        logger.info("Downloading %s from %s", filename, url)
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(response.text, encoding="utf-8")
        return pd.read_csv(io.StringIO(response.text))

    def _apply_crosswalk(self, df: pd.DataFrame, cols: list[str]) -> None:
        for col in cols:
            df[col] = normalize_series(df[col])

    def _coerce_results(self, raw: pd.DataFrame) -> pd.DataFrame:
        df = raw.rename(columns=_RESULTS_RENAME)[list(_RESULTS_RENAME.values())].copy()
        df["HOME_GOALS"] = pd.to_numeric(df["HOME_GOALS"], errors="coerce").astype("Int64")
        df["AWAY_GOALS"] = pd.to_numeric(df["AWAY_GOALS"], errors="coerce").astype("Int64")
        df["NEUTRAL"] = _parse_bool_column(df["NEUTRAL"])
        self._apply_crosswalk(df, ["HOME_TEAM", "AWAY_TEAM"])
        validate_results_df(df)
        return df

    def _coerce_shootouts(self, raw: pd.DataFrame) -> pd.DataFrame:
        df = raw.rename(columns=_SHOOTOUTS_RENAME)[list(_SHOOTOUTS_RENAME.values())].copy()
        self._apply_crosswalk(df, ["HOME_TEAM", "AWAY_TEAM"])
        validate_shootouts_df(df)
        return df

    def _coerce_goalscorers(self, raw: pd.DataFrame) -> pd.DataFrame:
        df = raw.rename(columns=_GOALSCORERS_RENAME)[list(_GOALSCORERS_RENAME.values())].copy()
        df["MINUTE"] = pd.to_numeric(df["MINUTE"], errors="coerce").astype("Int64")
        df["OWN_GOAL"] = _parse_bool_column(df["OWN_GOAL"])
        df["PENALTY"] = _parse_bool_column(df["PENALTY"])
        self._apply_crosswalk(df, ["HOME_TEAM", "AWAY_TEAM", "TEAM"])
        validate_goalscorers_df(df)
        return df


# ---------------------------------------------------------------------------
# Module-level factory functions (mirrors build_*_csv / generate_bracket_toml)
# ---------------------------------------------------------------------------


def load_martj42_results(config: Martj42Config) -> pd.DataFrame:
    """Load and return the martj42 results DataFrame."""
    return Martj42Loader(config=config).load_results()


def load_martj42_shootouts(config: Martj42Config) -> pd.DataFrame:
    """Load and return the martj42 shootouts DataFrame."""
    return Martj42Loader(config=config).load_shootouts()


def load_martj42_goalscorers(config: Martj42Config) -> pd.DataFrame:
    """Load and return the martj42 goalscorers DataFrame."""
    return Martj42Loader(config=config).load_goalscorers()
