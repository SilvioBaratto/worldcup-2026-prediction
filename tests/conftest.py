"""Shared test fixtures for the worldcup_playoff test suite."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from worldcup_playoff.config import AppConfig, BracketConfig, Matchup, SimulationConfig
from worldcup_playoff.simulation.distributions import FittedDistribution


# ---------------------------------------------------------------------------
# Feature columns matching train_data.csv schema
# ---------------------------------------------------------------------------

_FEATURE_COLS = [
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

_TEAMS = ["Brazil", "France", "Germany", "Argentina"]


@pytest.fixture
def sample_train_df() -> pd.DataFrame:
    """Minimal train_data.csv DataFrame with the correct schema.

    Contains enough rows per team to fit distributions (at least 10 per team
    in both home and away roles).
    """
    rng = np.random.default_rng(42)
    rows: list[dict[str, object]] = []

    # Generate matches between the four teams — each pair plays twice (home/away)
    matchups = [
        ("Brazil", "France"),
        ("Germany", "Argentina"),
        ("France", "Germany"),
        ("Argentina", "Brazil"),
        ("Brazil", "Germany"),
        ("France", "Argentina"),
        ("Germany", "Brazil"),
        ("Argentina", "France"),
        ("Brazil", "Argentina"),
        ("France", "Germany"),
        ("Germany", "France"),
        ("Argentina", "Brazil"),
    ]

    for home, away in matchups:
        home_goals = int(rng.integers(0, 4))
        away_goals = int(rng.integers(0, 4))
        # Ensure no draws so HOME_WIN is meaningful
        if home_goals == away_goals:
            home_goals += 1

        home_shots = home_goals * 5 + 7
        away_shots = away_goals * 5 + 7
        home_sot = max(home_goals + 2, home_shots // 3)
        away_sot = max(away_goals + 2, away_shots // 3)

        rows.append(
            {
                "HOME_TEAM": home,
                "AWAY_TEAM": away,
                "GOALS_home": home_goals,
                "SHOTS_home": home_shots,
                "SHOTS_ON_TARGET_home": home_sot,
                "POSSESSION_home": 50.0 + rng.uniform(-5, 5),
                "PASS_PCT_home": 75.0 + rng.uniform(-5, 5),
                "GOALS_away": away_goals,
                "SHOTS_away": away_shots,
                "SHOTS_ON_TARGET_away": away_sot,
                "POSSESSION_away": 50.0 + rng.uniform(-5, 5),
                "PASS_PCT_away": 75.0 + rng.uniform(-5, 5),
                "HOME_WIN": int(home_goals > away_goals),
            }
        )

    return pd.DataFrame(rows)


@pytest.fixture
def app_config() -> AppConfig:
    """Default AppConfig with no modifications."""
    return AppConfig()


@pytest.fixture
def sample_distributions() -> dict[str, list[FittedDistribution]]:
    """Pre-built distributions for four national teams.

    Five features each (goals, shots, shots-on-target, possession, pass-pct).
    """
    return {
        "Brazil": [
            FittedDistribution(name="norm", params=(1.8, 0.8)),
            FittedDistribution(name="norm", params=(16.0, 3.0)),
            FittedDistribution(name="norm", params=(5.0, 1.5)),
            FittedDistribution(name="norm", params=(52.0, 5.0)),
            FittedDistribution(name="norm", params=(77.0, 4.0)),
        ],
        "France": [
            FittedDistribution(name="norm", params=(1.6, 0.7)),
            FittedDistribution(name="norm", params=(15.0, 3.0)),
            FittedDistribution(name="norm", params=(4.8, 1.4)),
            FittedDistribution(name="norm", params=(51.0, 5.0)),
            FittedDistribution(name="norm", params=(76.0, 4.0)),
        ],
        "Germany": [
            FittedDistribution(name="norm", params=(1.7, 0.7)),
            FittedDistribution(name="norm", params=(15.5, 2.8)),
            FittedDistribution(name="norm", params=(4.9, 1.3)),
            FittedDistribution(name="norm", params=(50.0, 5.0)),
            FittedDistribution(name="norm", params=(76.5, 3.5)),
        ],
        "Argentina": [
            FittedDistribution(name="norm", params=(1.5, 0.7)),
            FittedDistribution(name="norm", params=(14.5, 2.5)),
            FittedDistribution(name="norm", params=(4.6, 1.2)),
            FittedDistribution(name="norm", params=(49.0, 5.0)),
            FittedDistribution(name="norm", params=(75.0, 3.5)),
        ],
    }


@pytest.fixture
def simple_bracket() -> list[Matchup]:
    """Minimal 4-matchup bracket (power of 2 — 8 teams)."""
    return [
        Matchup(home="Brazil", away="France"),
        Matchup(home="Germany", away="Argentina"),
        Matchup(home="Spain", away="England"),
        Matchup(home="Portugal", away="Netherlands"),
    ]


@pytest.fixture
def two_team_bracket() -> list[Matchup]:
    """Single matchup bracket (minimum valid bracket)."""
    return [Matchup(home="Brazil", away="France")]


@pytest.fixture
def bracket_config(simple_bracket: list[Matchup]) -> BracketConfig:
    """BracketConfig wrapping the 4-matchup simple_bracket."""
    return BracketConfig(name="Test World Cup 2026", matchups=simple_bracket)


@pytest.fixture
def mock_classifier() -> Any:
    """A fake classifier that always predicts home win (1)."""

    class _AlwaysHomeClassifier:
        def fit(self, X: Any, y: Any) -> "_AlwaysHomeClassifier":
            return self

        def predict(self, X: np.ndarray) -> np.ndarray:
            return np.ones(X.shape[0], dtype=int)

    return _AlwaysHomeClassifier()


@pytest.fixture
def tmp_project_root(tmp_path: Path) -> Path:
    """A temporary directory tree that mirrors the project layout."""
    (tmp_path / "dataset" / "csv").mkdir(parents=True)
    (tmp_path / "dataset").mkdir(parents=True, exist_ok=True)
    (tmp_path / "output" / "models").mkdir(parents=True)
    (tmp_path / "config").mkdir(parents=True)
    return tmp_path
