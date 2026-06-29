"""WC2026 squad market values — a player-level strength signal.

Total squad market value (Transfermarkt "wisdom of the crowd") per national team,
in millions of euros, as published for the 2026 World Cup. Used as an optional
prior alongside Elo: market value captures the *current* squad quality (new
generations, who is actually available) that a results-only history misses, and
is consistently one of the strongest single predictors in the academic World Cup
forecasting literature (Groll, Ley, Schauberger & Van Eetvelde, 2019).

Keys use the exact martj42 team spellings so the values join the fitted abilities
directly. Snapshot — values drift; refresh per tournament.
"""

from __future__ import annotations

# team -> total squad market value (EUR millions)
WC2026_SQUAD_VALUE_EUR_M: dict[str, float] = {
    "France": 1520.0,
    "England": 1360.0,
    "Spain": 1220.0,
    "Portugal": 1010.0,
    "Germany": 947.0,
    "Brazil": 928.2,
    "Argentina": 807.5,
    "Netherlands": 754.2,
    "Norway": 589.9,
    "Belgium": 547.5,
    "Ivory Coast": 522.1,
    "Senegal": 478.1,
    "Turkey": 473.7,
    "Morocco": 447.7,
    "Sweden": 406.08,
    "Croatia": 387.3,
    "United States": 385.6,
    "Ecuador": 368.7,
    "Uruguay": 359.3,
    "Switzerland": 332.5,
    "Colombia": 302.35,
    "Japan": 270.85,
    "Algeria": 256.9,
    "Austria": 245.2,
    "Ghana": 234.5,
    "Canada": 198.65,
    "Mexico": 191.85,
    "Czech Republic": 188.18,
    "Scotland": 170.25,
    "Paraguay": 153.65,
    "Bosnia and Herzegovina": 146.4,
    "DR Congo": 143.9,
    "South Korea": 139.05,
    "Egypt": 116.48,
    "Uzbekistan": 85.33,
    "Australia": 77.45,
    "Tunisia": 69.95,
    "Haiti": 55.9,
    "Cabo Verde": 49.25,
    "South Africa": 49.25,
    "Saudi Arabia": 40.68,
    "Panama": 34.55,
    "New Zealand": 34.45,
    "Iran": 32.05,
    "Curaçao": 25.78,
    "Iraq": 21.2,
    "Jordan": 20.3,
    "Qatar": 19.93,
}

__all__ = ["WC2026_SQUAD_VALUE_EUR_M"]
