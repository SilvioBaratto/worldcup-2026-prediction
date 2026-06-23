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

    @field_validator("n_simulations")
    @classmethod
    def n_simulations_must_be_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("n_simulations must be at least 1")
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


class AppConfig(BaseModel):
    data: DataConfig = DataConfig()
    features: FeaturesConfig = FeaturesConfig()
    training: TrainingConfig = TrainingConfig()
    distributions: DistributionConfig = DistributionConfig()
    simulation: SimulationConfig = SimulationConfig()
    visualization: VisualizationConfig = VisualizationConfig()
    client: ClientConfig = ClientConfig()
    martj42: Martj42Config = Martj42Config()
    live: LiveConfig = LiveConfig()


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
