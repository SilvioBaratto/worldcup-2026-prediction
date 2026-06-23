"""Team-name normalization: maps alias spellings to canonical country names."""

from __future__ import annotations

import logging
import re

import pandas as pd

logger = logging.getLogger(__name__)

# Alias → canonical mapping. Keys are stored in their natural casing;
# lookup normalizes to lowercase+stripped at call time so the dict stays compact.
CANONICAL_NAMES: dict[str, str] = {
    # Turkey
    "Türkiye": "Turkey",
    "Turkey": "Turkey",
    # Czech Republic
    "Czechia": "Czech Republic",
    "Czech Republic": "Czech Republic",
    # South Korea
    "South Korea": "South Korea",
    "Korea Republic": "South Korea",
    "Republic of Korea": "South Korea",
    # United States
    "United States": "United States",
    "USA": "United States",
    "United States of America": "United States",
    # Iran
    "IR Iran": "Iran",
    "Iran": "Iran",
    # China
    "China PR": "China",
    "China": "China",
    # DRC / Congo
    "DR Congo": "DR Congo",
    "Congo DR": "DR Congo",
    # Ivory Coast
    "Ivory Coast": "Ivory Coast",
    "Côte d'Ivoire": "Ivory Coast",
    # North Macedonia
    "North Macedonia": "North Macedonia",
    "Macedonia": "North Macedonia",
    # Bosnia
    "Bosnia and Herzegovina": "Bosnia and Herzegovina",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    # Trinidad and Tobago
    "Trinidad and Tobago": "Trinidad and Tobago",
    "Trinidad & Tobago": "Trinidad and Tobago",
    # Cabo Verde
    "Cabo Verde": "Cabo Verde",
    "Cape Verde": "Cabo Verde",
}

# Reverse lookup: lowercased+stripped alias → canonical, built once at import time.
_LOOKUP: dict[str, str] = {
    re.sub(r"\s+", " ", k.strip()).lower(): v for k, v in CANONICAL_NAMES.items()
}


def normalize_team(name: str) -> str:
    """Return the canonical country name for *name*, or the cleaned input if unknown."""
    cleaned = re.sub(r"\s+", " ", name.strip())
    return _LOOKUP.get(cleaned.lower(), cleaned)


def normalize_series(s: pd.Series) -> pd.Series:
    """Vectorized version of *normalize_team*; preserves index and length."""
    return s.map(normalize_team)


def is_known(name: str) -> bool:
    """Return True when *name* maps to a canonical country name."""
    return re.sub(r"\s+", " ", name.strip()).lower() in _LOOKUP
