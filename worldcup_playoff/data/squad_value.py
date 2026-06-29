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

# Historical total squad market value (EUR millions) *as of each tournament*,
# from the published Transfermarkt-based rankings (planetfootball 2022; Goal.com
# 2018). Used to validate the market-value prior on past World Cups. Coverage is
# partial for the cheapest 2018 outsiders (omitted rather than guessed); the
# blend simply leaves an unlisted team on its Elo-only abilities. 2014 is left
# out — no complete, verifiable as-of-2014 source was available.
WC_SQUAD_VALUE_HISTORICAL_EUR_M: dict[int, dict[str, float]] = {
    2018: {
        "France": 1037.0, "Spain": 1001.0, "Brazil": 948.0, "Germany": 896.0,
        "England": 668.0, "Belgium": 602.0, "Argentina": 529.0, "Portugal": 355.0,
        "Croatia": 291.0, "Uruguay": 280.0, "Senegal": 252.0, "Denmark": 200.0,
        "Poland": 200.0, "Colombia": 196.0, "Serbia": 181.0, "Switzerland": 161.0,
        "Egypt": 150.0, "Russia": 105.0, "Mexico": 101.0, "Nigeria": 100.0,
        "Morocco": 90.0, "Sweden": 86.0, "South Korea": 63.0, "Iceland": 56.0,
        "Japan": 53.0,
    },
    2022: {
        "England": 1260.0, "Brazil": 1140.0, "France": 1060.0, "Portugal": 937.0,
        "Spain": 902.0, "Germany": 855.5, "Argentina": 633.2, "Netherlands": 587.25,
        "Belgium": 563.2, "Uruguay": 449.7, "Croatia": 377.0, "Serbia": 359.5,
        "Denmark": 353.0, "Senegal": 288.0, "Switzerland": 281.0, "United States": 277.4,
        "Poland": 255.6, "Morocco": 251.1, "Ghana": 216.9, "Canada": 187.3,
        "Mexico": 176.1, "South Korea": 164.48, "Wales": 160.15, "Cameroon": 155.0,
        "Japan": 154.0, "Ecuador": 146.5, "Tunisia": 62.4, "Iran": 59.53,
        "Australia": 38.4, "Saudi Arabia": 25.2, "Costa Rica": 18.75, "Qatar": 14.9,
    },
}

__all__ = ["WC2026_SQUAD_VALUE_EUR_M", "WC_SQUAD_VALUE_HISTORICAL_EUR_M"]
