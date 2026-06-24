"""Configuration loading and validation via Pydantic models."""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator


class Martj42Config(BaseModel):
    """Configuration for the no-key martj42 CC0 historical results loader."""

    model_config = ConfigDict(extra="ignore")

    base_url: str = "https://raw.githubusercontent.com/martj42/international_results/master/"
    cache_dir: Path = Path("dataset/martj42")
    results_file: str = "results.csv"
    shootouts_file: str = "shootouts.csv"
    goalscorers_file: str = "goalscorers.csv"


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    matches_path: str | None = "dataset/csv/matches.csv"
    teams_path: str | None = "dataset/csv/teams.csv"
    output_path: str = "dataset/train_data.csv"
    min_date: str = "2006-01-01"
    train_cutoff_date: str = "2026-06-01"
    epsilon: float = 0.001
    matches_start_year: int = 2006
    matches_end_year: int = 2026
    ranking_csv_path: str = "dataset/csv/ranking.csv"
    ranking_start_year: int = 2006
    ranking_end_year: int = 2026
    players_csv_path: str = "dataset/csv/players.csv"
    players_competition: str = "WC"
    match_details_csv_path: str = "dataset/csv/match_details.csv"
    match_details_checkpoint_every: int = 100


class FeaturesConfig(BaseModel):
    """Per-team match statistics fed to the classifier.

    Five features per team (home + away) mirroring the NBA layout, expressed as
    football box-score metrics: goals, shots, shots on target, possession %, and
    pass accuracy %.
    """

    selected: list[str] = [
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
    per_team_count: int = 5


class SVMConfig(BaseModel):
    C: float = 0.1
    gamma: float = 0.1
    kernel: str = "linear"


class RandomForestConfig(BaseModel):
    n_estimators: int = 500
    max_features: str = "sqrt"
    max_depth: int = 50
    bootstrap: bool = True


class NaiveBayesConfig(BaseModel):
    var_smoothing: float = 1.873817422860383e-07


class TrainingConfig(BaseModel):
    test_size: float = 0.3
    random_state: int = 42
    svm: SVMConfig = SVMConfig()
    random_forest: RandomForestConfig = RandomForestConfig()
    naive_bayes: NaiveBayesConfig = NaiveBayesConfig()


class DistributionConfig(BaseModel):
    min_season: int = 2018
    candidates: list[str] = [
        "norm",
        "t",
        "f",
        "chi",
        "cosine",
        "alpha",
        "beta",
        "gamma",
        "dgamma",
        "dweibull",
        "maxwell",
        "pareto",
        "fisk",
    ]


class SimulationConfig(BaseModel):
    """Knockout simulation settings.

    Unlike the NBA best-of-7 series, World Cup knockout ties are decided by a
    single match (extra time / penalties collapse into one win/loss outcome), so
    there is no series length to configure.
    """

    n_simulations: int = 10000
    classifier: str = "naive_bayes"
    extra_time_factor: float = 0.33
    random_seed: int = 42

    @field_validator("n_simulations")
    @classmethod
    def n_simulations_must_be_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("n_simulations must be at least 1")
        return v

    @field_validator("extra_time_factor")
    @classmethod
    def extra_time_factor_in_range(cls, v: float) -> float:
        if not (0.0 < v <= 1.0):
            raise ValueError("extra_time_factor must be in (0.0, 1.0]")
        return v

    @field_validator("random_seed")
    @classmethod
    def random_seed_must_be_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("random_seed must be >= 0")
        return v


class VisualizationConfig(BaseModel):
    dpi: int = 80
    style: str = "seaborn-v0_8"
    output_dir: str = "output/plots"


class ClientConfig(BaseModel):
    # football-data.org free tier allows 10 requests/minute -> 6s spacing.
    delay: float = 6.0
    max_retries: int = 5
    backoff_base: float = 2.0
    timeout: int = 120
    use_custom_headers: bool = True

    @field_validator("delay", "backoff_base")
    @classmethod
    def must_be_positive_float(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("must be positive")
        return v

    @field_validator("max_retries")
    @classmethod
    def must_be_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("must be >= 0")
        return v

    @field_validator("timeout")
    @classmethod
    def timeout_must_be_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("must be >= 1")
        return v


class LiveConfig(BaseModel):
    """Configuration for the live WC2026 football-data.org adapter."""

    model_config = ConfigDict(extra="ignore")

    competition: str = "WC"


class EloConfig(BaseModel):
    """Configuration for the World Football Elo engine."""

    model_config = ConfigDict(extra="ignore")

    initial_rating: float = 1500.0
    home_advantage: float = 100.0
    k_friendly: int = 20
    k_qualifier: int = 30
    k_continental: int = 40
    k_world_cup: int = 60
    # Checked in order: qualifier → continental → world_cup → friendly.
    # qualifier_keywords checked before world_cup_keywords so that
    # "FIFA World Cup qualification" maps to qualifier tier, not world-cup.
    qualifier_keywords: list[str] = ["qualification", "qualifying", "qualifier"]
    continental_keywords: list[str] = [
        "Copa América",           # martj42 exact; accent-normalised match also catches "Copa America"
        "UEFA Euro",              # martj42 uses "UEFA Euro" / "UEFA Euro qualification"
        "African Cup of Nations", # martj42 exact
        "Asian Cup",              # substring of "AFC Asian Cup"
        "Gold Cup",               # martj42 exact
        "Nations League",         # covers "UEFA Nations League", "CONCACAF Nations League"
    ]
    world_cup_keywords: list[str] = ["World Cup"]

    @field_validator("initial_rating")
    @classmethod
    def initial_rating_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("initial_rating must be positive")
        return v

    @field_validator("home_advantage")
    @classmethod
    def home_advantage_must_be_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("home_advantage must be non-negative")
        return v

    @field_validator("k_friendly", "k_qualifier", "k_continental", "k_world_cup")
    @classmethod
    def k_factor_must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("K factor must be positive")
        return v


class PoissonConfig(BaseModel):
    """Configuration for the Dixon-Coles bivariate-Poisson estimator.

    All time-decay values are in DAYS.
    """

    model_config = ConfigDict(extra="ignore")

    # Exponential decay half-life in DAYS (recent matches weigh more).
    half_life_days: float = 365.0
    max_goals: int = 10
    rho_init: float = -0.1
    home_adv_init: float = 0.25
    random_seed: int = 42
    optimizer_maxiter: int = 200

    @field_validator("half_life_days")
    @classmethod
    def half_life_days_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("half_life_days must be positive")
        return v

    @field_validator("max_goals")
    @classmethod
    def max_goals_must_be_at_least_one(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_goals must be at least 1")
        return v

    @field_validator("rho_init")
    @classmethod
    def rho_init_must_be_at_most_zero(cls, v: float) -> float:
        if not (-0.99 <= v <= 0):
            raise ValueError("rho_init must be in [-0.99, 0]")
        return v

    @field_validator("optimizer_maxiter")
    @classmethod
    def optimizer_maxiter_must_be_at_least_one(cls, v: int) -> int:
        if v < 1:
            raise ValueError("optimizer_maxiter must be at least 1")
        return v


class FeatureBuildConfig(BaseModel):
    """Configuration for the football-only feature assembler (Cycle 3+)."""

    model_config = ConfigDict(extra="ignore")

    ranking_staleness_cutoff: str = "2020-12-10"
    form_window: int = 5
    form_half_life_days: float = 365.0
    random_seed: int = 42
    confederation_fallback: bool = True

    @field_validator("form_window")
    @classmethod
    def form_window_must_be_at_least_one(cls, v: int) -> int:
        if v < 1:
            raise ValueError("form_window must be >= 1")
        return v

    @field_validator("form_half_life_days")
    @classmethod
    def form_half_life_days_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("form_half_life_days must be positive")
        return v


class HybridConfig(BaseModel):
    """Configuration for the Groll-style RF/GBM goal-based hybrid model."""

    model_config = ConfigDict(extra="ignore")

    rf_n_estimators: int = 300
    rf_max_depth: int | None = None
    gb_n_estimators: int = 200
    gb_learning_rate: float = 0.05
    max_goals: int = 10
    rho: float = -0.1
    test_size: float = 0.2
    random_seed: int = 42

    @field_validator("test_size")
    @classmethod
    def test_size_must_be_in_open_unit_interval(cls, v: float) -> float:
        if not (0.0 < v < 1.0):
            raise ValueError("test_size must be strictly between 0 and 1")
        return v


class OrderedLogitConfig(BaseModel):
    """Configuration for the Elo-diff ordered logit secondary/fallback model."""

    model_config = ConfigDict(extra="ignore")

    features: list[str] = ["elo_diff"]
    maxiter: int = 100
    test_size: float = 0.2
    random_seed: int = 42

    @field_validator("test_size")
    @classmethod
    def test_size_must_be_in_open_unit_interval(cls, v: float) -> float:
        if not (0.0 < v < 1.0):
            raise ValueError("test_size must be strictly between 0 and 1")
        return v


class OddsConfig(BaseModel):
    """Configuration for the historical bookmaker odds scraper (backtest baseline only)."""

    model_config = ConfigDict(extra="ignore")

    cache_dir: Path = Path("dataset/odds")
    seasons: list[int] = [2014, 2018, 2022]
    source_url: str = "https://www.oddsportal.com"
    request_timeout: int = 30
    user_agent: str = "Mozilla/5.0 (compatible; worldcup-playoff/1.0)"
    enabled: bool = True
    fallback_to_cache: bool = True

    @field_validator("request_timeout")
    @classmethod
    def request_timeout_must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("request_timeout must be positive")
        return v

    markets: list[str] = ["outright", "match"]
    match_url_template: str = ""

    @field_validator("seasons")
    @classmethod
    def seasons_must_be_non_empty(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("seasons must not be empty")
        return v

    @field_validator("markets")
    @classmethod
    def markets_must_be_non_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("markets must not be empty")
        return v


class RFConfig(BaseModel):
    """Configuration for the Groll-hybrid Random-Forest tuning surface (distinct from legacy RF)."""

    model_config = ConfigDict(extra="ignore")

    n_estimators: int = 300
    max_depth: int | None = None
    min_samples_leaf: int = 1
    test_size: float = 0.2
    random_seed: int = 42

    @field_validator("n_estimators")
    @classmethod
    def n_estimators_must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("n_estimators must be positive")
        return v

    @field_validator("test_size")
    @classmethod
    def test_size_in_unit_interval(cls, v: float) -> float:
        if not (0.0 < v < 1.0):
            raise ValueError("test_size must be strictly between 0 and 1")
        return v


class AppConfig(BaseModel):
    data: DataConfig = DataConfig()
    features: FeaturesConfig = FeaturesConfig()
    features_build: FeatureBuildConfig = FeatureBuildConfig()
    training: TrainingConfig = TrainingConfig()
    distributions: DistributionConfig = DistributionConfig()
    simulation: SimulationConfig = SimulationConfig()
    visualization: VisualizationConfig = VisualizationConfig()
    client: ClientConfig = ClientConfig()
    martj42: Martj42Config = Martj42Config()
    live: LiveConfig = LiveConfig()
    elo: EloConfig = EloConfig()
    poisson: PoissonConfig = PoissonConfig()
    hybrid: HybridConfig = HybridConfig()
    ordered_logit: OrderedLogitConfig = OrderedLogitConfig()
    odds: OddsConfig = OddsConfig()
    rf: RFConfig = RFConfig()


class Matchup(BaseModel):
    home: str
    away: str
    group: str = ""


class BracketConfig(BaseModel):
    name: str = ""
    matchups: list[Matchup] = []


def load_config(path: Path) -> AppConfig:
    """Load pipeline configuration from a TOML file."""
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return AppConfig.model_validate(raw)


def load_bracket(path: Path) -> BracketConfig:
    """Load bracket definition from a TOML file."""
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return BracketConfig.model_validate(raw)
