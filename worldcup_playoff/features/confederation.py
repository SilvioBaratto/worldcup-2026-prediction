"""Static confederation map and FIFA-ranking fallback resolver.

All 48 WC2026 qualified nations plus common historical nations are mapped to
one of the six FIFA confederations. Keys are crosswalk-canonical country names
so ``normalize_team(key) == key`` holds for every entry.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass

from worldcup_playoff.data.crosswalk import normalize_team

logger = logging.getLogger(__name__)

# Fixed, documented order used for one-hot encoding in downstream assemblers.
CONFEDERATIONS: tuple[str, ...] = ("UEFA", "CONMEBOL", "CAF", "AFC", "CONCACAF", "OFC")

# ---------------------------------------------------------------------------
# Confederation map — keys MUST be crosswalk-canonical (normalize_team(k)==k)
# ---------------------------------------------------------------------------

CONFEDERATION_MAP: dict[str, str] = {
    # ── UEFA ──────────────────────────────────────────────────────────────
    "Albania": "UEFA",
    "Andorra": "UEFA",
    "Armenia": "UEFA",
    "Austria": "UEFA",
    "Azerbaijan": "UEFA",
    "Belarus": "UEFA",
    "Belgium": "UEFA",
    "Bosnia and Herzegovina": "UEFA",
    "Bulgaria": "UEFA",
    "Croatia": "UEFA",
    "Cyprus": "UEFA",
    "Czech Republic": "UEFA",
    "Denmark": "UEFA",
    "England": "UEFA",
    "Estonia": "UEFA",
    "Faroe Islands": "UEFA",
    "Finland": "UEFA",
    "France": "UEFA",
    "Georgia": "UEFA",
    "Germany": "UEFA",
    "Gibraltar": "UEFA",
    "Greece": "UEFA",
    "Hungary": "UEFA",
    "Iceland": "UEFA",
    "Israel": "UEFA",
    "Italy": "UEFA",
    "Kazakhstan": "UEFA",
    "Kosovo": "UEFA",
    "Latvia": "UEFA",
    "Liechtenstein": "UEFA",
    "Lithuania": "UEFA",
    "Luxembourg": "UEFA",
    "Malta": "UEFA",
    "Moldova": "UEFA",
    "Montenegro": "UEFA",
    "Netherlands": "UEFA",
    "North Macedonia": "UEFA",
    "Northern Ireland": "UEFA",
    "Norway": "UEFA",
    "Poland": "UEFA",
    "Portugal": "UEFA",
    "Republic of Ireland": "UEFA",
    "Romania": "UEFA",
    "Russia": "UEFA",
    "San Marino": "UEFA",
    "Scotland": "UEFA",
    "Serbia": "UEFA",
    "Slovakia": "UEFA",
    "Slovenia": "UEFA",
    "Spain": "UEFA",
    "Sweden": "UEFA",
    "Switzerland": "UEFA",
    "Turkey": "UEFA",
    "Ukraine": "UEFA",
    "Wales": "UEFA",
    # ── CONMEBOL ──────────────────────────────────────────────────────────
    "Argentina": "CONMEBOL",
    "Bolivia": "CONMEBOL",
    "Brazil": "CONMEBOL",
    "Chile": "CONMEBOL",
    "Colombia": "CONMEBOL",
    "Ecuador": "CONMEBOL",
    "Paraguay": "CONMEBOL",
    "Peru": "CONMEBOL",
    "Uruguay": "CONMEBOL",
    "Venezuela": "CONMEBOL",
    # ── CAF ───────────────────────────────────────────────────────────────
    "Algeria": "CAF",
    "Angola": "CAF",
    "Benin": "CAF",
    "Botswana": "CAF",
    "Burkina Faso": "CAF",
    "Burundi": "CAF",
    "Cabo Verde": "CAF",
    "Cameroon": "CAF",
    "Central African Republic": "CAF",
    "Chad": "CAF",
    "Comoros": "CAF",
    "Congo": "CAF",
    "DR Congo": "CAF",
    "Djibouti": "CAF",
    "Egypt": "CAF",
    "Equatorial Guinea": "CAF",
    "Eritrea": "CAF",
    "Eswatini": "CAF",
    "Ethiopia": "CAF",
    "Gabon": "CAF",
    "Gambia": "CAF",
    "Ghana": "CAF",
    "Guinea": "CAF",
    "Guinea-Bissau": "CAF",
    "Ivory Coast": "CAF",
    "Kenya": "CAF",
    "Lesotho": "CAF",
    "Liberia": "CAF",
    "Libya": "CAF",
    "Madagascar": "CAF",
    "Malawi": "CAF",
    "Mali": "CAF",
    "Mauritania": "CAF",
    "Mauritius": "CAF",
    "Morocco": "CAF",
    "Mozambique": "CAF",
    "Namibia": "CAF",
    "Niger": "CAF",
    "Nigeria": "CAF",
    "Rwanda": "CAF",
    "Sao Tome and Principe": "CAF",
    "Senegal": "CAF",
    "Seychelles": "CAF",
    "Sierra Leone": "CAF",
    "Somalia": "CAF",
    "South Africa": "CAF",
    "South Sudan": "CAF",
    "Sudan": "CAF",
    "Tanzania": "CAF",
    "Togo": "CAF",
    "Tunisia": "CAF",
    "Uganda": "CAF",
    "Zambia": "CAF",
    "Zimbabwe": "CAF",
    # ── AFC ───────────────────────────────────────────────────────────────
    "Afghanistan": "AFC",
    "Australia": "AFC",
    "Bahrain": "AFC",
    "Bangladesh": "AFC",
    "Bhutan": "AFC",
    "Brunei": "AFC",
    "Cambodia": "AFC",
    "China": "AFC",
    "Chinese Taipei": "AFC",
    "Guam": "AFC",
    "Hong Kong": "AFC",
    "India": "AFC",
    "Indonesia": "AFC",
    "Iran": "AFC",
    "Iraq": "AFC",
    "Japan": "AFC",
    "Jordan": "AFC",
    "Kuwait": "AFC",
    "Kyrgyzstan": "AFC",
    "Laos": "AFC",
    "Lebanon": "AFC",
    "Macao": "AFC",
    "Malaysia": "AFC",
    "Maldives": "AFC",
    "Mongolia": "AFC",
    "Myanmar": "AFC",
    "Nepal": "AFC",
    "North Korea": "AFC",
    "Oman": "AFC",
    "Pakistan": "AFC",
    "Palestine": "AFC",
    "Philippines": "AFC",
    "Qatar": "AFC",
    "Saudi Arabia": "AFC",
    "Singapore": "AFC",
    "South Korea": "AFC",
    "Sri Lanka": "AFC",
    "Syria": "AFC",
    "Tajikistan": "AFC",
    "Thailand": "AFC",
    "Timor-Leste": "AFC",
    "Turkmenistan": "AFC",
    "United Arab Emirates": "AFC",
    "Uzbekistan": "AFC",
    "Vietnam": "AFC",
    "Yemen": "AFC",
    # ── CONCACAF ──────────────────────────────────────────────────────────
    "Anguilla": "CONCACAF",
    "Antigua and Barbuda": "CONCACAF",
    "Aruba": "CONCACAF",
    "Bahamas": "CONCACAF",
    "Barbados": "CONCACAF",
    "Belize": "CONCACAF",
    "Bermuda": "CONCACAF",
    "British Virgin Islands": "CONCACAF",
    "Canada": "CONCACAF",
    "Cayman Islands": "CONCACAF",
    "Costa Rica": "CONCACAF",
    "Cuba": "CONCACAF",
    "Curacao": "CONCACAF",
    "Dominica": "CONCACAF",
    "Dominican Republic": "CONCACAF",
    "El Salvador": "CONCACAF",
    "Grenada": "CONCACAF",
    "Guatemala": "CONCACAF",
    "Guyana": "CONCACAF",
    "Haiti": "CONCACAF",
    "Honduras": "CONCACAF",
    "Jamaica": "CONCACAF",
    "Mexico": "CONCACAF",
    "Montserrat": "CONCACAF",
    "Nicaragua": "CONCACAF",
    "Panama": "CONCACAF",
    "Puerto Rico": "CONCACAF",
    "Saint Kitts and Nevis": "CONCACAF",
    "Saint Lucia": "CONCACAF",
    "Saint Vincent and the Grenadines": "CONCACAF",
    "Suriname": "CONCACAF",
    "Trinidad and Tobago": "CONCACAF",
    "Turks and Caicos Islands": "CONCACAF",
    "United States": "CONCACAF",
    "US Virgin Islands": "CONCACAF",
    # ── OFC ───────────────────────────────────────────────────────────────
    "American Samoa": "OFC",
    "Cook Islands": "OFC",
    "Fiji": "OFC",
    "New Caledonia": "OFC",
    "New Zealand": "OFC",
    "Papua New Guinea": "OFC",
    "Samoa": "OFC",
    "Solomon Islands": "OFC",
    "Tahiti": "OFC",
    "Tonga": "OFC",
    "Vanuatu": "OFC",
}


# ---------------------------------------------------------------------------
# Value object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RankingResolution:
    """Result of a FIFA-ranking lookup with confederation fallback."""

    value: float | None
    confederation: str | None
    used_fallback: bool


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def confederation_of(team: str) -> str | None:
    """Return the confederation code for *team*, or ``None`` when unknown."""
    return CONFEDERATION_MAP.get(normalize_team(team))


def _is_stale(as_of: str, staleness_cutoff: str) -> bool:
    """Return True when the ranking snapshot date exceeds the cutoff."""
    return as_of > staleness_cutoff


def resolve_ranking(
    team: str,
    ranking: Mapping[str, float] | None,
    as_of: str,
    staleness_cutoff: str,
    *,
    confed_map: dict[str, str] = CONFEDERATION_MAP,
) -> RankingResolution:
    """Return ranking value when fresh and present, else fall back to confederation.

    Staleness: *as_of* (match date) > *staleness_cutoff* means the ranking data
    predates this match by more than the configured window — treat as stale.
    """
    canonical = normalize_team(team)
    confederation = confed_map.get(canonical)

    if ranking is None or _is_stale(as_of, staleness_cutoff) or canonical not in ranking:
        return RankingResolution(value=None, confederation=confederation, used_fallback=True)

    return RankingResolution(
        value=ranking[canonical],
        confederation=confederation,
        used_fallback=False,
    )
