"""Data layer for the WC2026 forecast: martj42 historical results (CC0), Elo
ratings, the live tournament-state adapter, the official R32 bracket rules, the
football-data.org client, team-name crosswalk, and the backtest odds scraper."""

from worldcup_playoff.data.elo import (
    EloEngine,
    EloRating,
    EloResult,
    MatchEloDiff,
    compute_elo,
    seed_wc2026,
)
from worldcup_playoff.data.wc2026_bracket import (
    GROUPS,
    R32_SLOTS,
    THIRD_PLACE_COMBINATIONS,
    assign_thirds,
    rank_third_places,
    resolve_r32,
)
from worldcup_playoff.data.live import (
    GroupStanding,
    LiveMatch,
    LiveTournamentAdapter,
    TournamentState,
    fetch_tournament_state,
)
from worldcup_playoff.data.client import CUSTOM_HEADERS, FootballClient
from worldcup_playoff.data.crosswalk import CANONICAL_NAMES, normalize_series, normalize_team
from worldcup_playoff.data.odds import OddsScraper, de_vig, load_odds
from worldcup_playoff.data.martj42_loader import (
    Martj42Loader,
    REQUIRED_MARTJ42_GOALSCORERS_COLUMNS,
    REQUIRED_MARTJ42_RESULTS_COLUMNS,
    REQUIRED_MARTJ42_SHOOTOUTS_COLUMNS,
    load_martj42_goalscorers,
    load_martj42_results,
    load_martj42_shootouts,
    validate_goalscorers_df,
    validate_results_df,
    validate_shootouts_df,
    wc2026_schedule,
)

__all__ = [
    "EloEngine",
    "EloRating",
    "EloResult",
    "MatchEloDiff",
    "compute_elo",
    "seed_wc2026",
    "GROUPS",
    "R32_SLOTS",
    "THIRD_PLACE_COMBINATIONS",
    "assign_thirds",
    "rank_third_places",
    "resolve_r32",
    "GroupStanding",
    "LiveMatch",
    "LiveTournamentAdapter",
    "TournamentState",
    "fetch_tournament_state",
    "CANONICAL_NAMES",
    "CUSTOM_HEADERS",
    "FootballClient",
    "Martj42Loader",
    "REQUIRED_MARTJ42_GOALSCORERS_COLUMNS",
    "REQUIRED_MARTJ42_RESULTS_COLUMNS",
    "REQUIRED_MARTJ42_SHOOTOUTS_COLUMNS",
    "load_martj42_goalscorers",
    "load_martj42_results",
    "load_martj42_shootouts",
    "OddsScraper",
    "de_vig",
    "load_odds",
    "normalize_series",
    "normalize_team",
    "validate_goalscorers_df",
    "validate_results_df",
    "validate_shootouts_df",
    "wc2026_schedule",
]
