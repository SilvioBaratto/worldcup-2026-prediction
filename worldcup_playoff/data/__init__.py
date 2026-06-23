"""Data loading and API client utilities for the World Cup prediction pipeline."""

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
from worldcup_playoff.data.bracket_builder import (
    BracketBuilder,
    generate_bracket_toml,
)
from worldcup_playoff.data.builders import (
    MatchDetailsBuilder,
    MatchesBuilder,
    PlayersBuilder,
    RankingBuilder,
    TeamsBuilder,
    build_match_details_csv,
    build_matches_csv,
    build_players_csv,
    build_ranking_csv,
    build_teams_csv,
)
from worldcup_playoff.data.client import CUSTOM_HEADERS, FootballClient
from worldcup_playoff.data.cleaner import DataCleaner
from worldcup_playoff.data.crosswalk import CANONICAL_NAMES, normalize_series, normalize_team
from worldcup_playoff.data.loader import (
    DataLoader,
    REQUIRED_MATCHES_COLUMNS,
    REQUIRED_TEAMS_COLUMNS,
    validate_matches_df,
    validate_teams_df,
)
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
    "GROUPS",
    "R32_SLOTS",
    "THIRD_PLACE_COMBINATIONS",
    "assign_thirds",
    "rank_third_places",
    "resolve_r32",
    "BracketBuilder",
    "GroupStanding",
    "LiveMatch",
    "LiveTournamentAdapter",
    "TournamentState",
    "fetch_tournament_state",
    "CANONICAL_NAMES",
    "CUSTOM_HEADERS",
    "DataCleaner",
    "DataLoader",
    "FootballClient",
    "Martj42Loader",
    "MatchDetailsBuilder",
    "MatchesBuilder",
    "PlayersBuilder",
    "RankingBuilder",
    "REQUIRED_MARTJ42_GOALSCORERS_COLUMNS",
    "REQUIRED_MARTJ42_RESULTS_COLUMNS",
    "REQUIRED_MARTJ42_SHOOTOUTS_COLUMNS",
    "REQUIRED_MATCHES_COLUMNS",
    "REQUIRED_TEAMS_COLUMNS",
    "TeamsBuilder",
    "build_match_details_csv",
    "build_matches_csv",
    "build_players_csv",
    "build_ranking_csv",
    "build_teams_csv",
    "generate_bracket_toml",
    "load_martj42_goalscorers",
    "load_martj42_results",
    "load_martj42_shootouts",
    "normalize_series",
    "normalize_team",
    "validate_goalscorers_df",
    "validate_matches_df",
    "validate_results_df",
    "validate_shootouts_df",
    "validate_teams_df",
    "wc2026_schedule",
]
