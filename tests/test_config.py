"""Tests for config loading and Pydantic validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from worldcup_playoff.config import (
    AppConfig,
    BracketConfig,
    ClientConfig,
    DataConfig,
    DistributionConfig,
    Matchup,
    SimulationConfig,
    load_bracket,
    load_config,
)


# ---------------------------------------------------------------------------
# SimulationConfig validation — n_simulations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [1, 100, 10000])
def test_valid_n_simulations_is_accepted(value: int) -> None:
    config = SimulationConfig(n_simulations=value)
    assert config.n_simulations == value


def test_n_simulations_zero_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        SimulationConfig(n_simulations=0)


def test_n_simulations_negative_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        SimulationConfig(n_simulations=-1)


def test_simulation_config_has_no_series_fields() -> None:
    """World Cup uses single-match ties — no series-length fields on SimulationConfig."""
    cfg = SimulationConfig()
    assert not hasattr(cfg, "games_per_series")
    assert not hasattr(cfg, "wins_to_advance")


def test_simulation_config_default_classifier() -> None:
    assert SimulationConfig().classifier == "naive_bayes"


# ---------------------------------------------------------------------------
# DataConfig
# ---------------------------------------------------------------------------


def test_matches_path_default_is_csv_subdir() -> None:
    assert DataConfig().matches_path == "dataset/csv/matches.csv"


def test_teams_path_default_is_csv_subdir() -> None:
    assert DataConfig().teams_path == "dataset/csv/teams.csv"


def test_matches_path_can_be_none() -> None:
    assert DataConfig(matches_path=None).matches_path is None


def test_teams_path_can_be_none() -> None:
    assert DataConfig(teams_path=None).teams_path is None


def test_extra_fields_ignored() -> None:
    cfg = DataConfig(games_path="whatever")  # type: ignore[call-arg]
    assert cfg.matches_path == "dataset/csv/matches.csv"


def test_output_path_default() -> None:
    assert DataConfig().output_path == "dataset/train_data.csv"


def test_epsilon_default() -> None:
    assert DataConfig().epsilon == 0.001


def test_min_date_default() -> None:
    assert DataConfig().min_date == "2006-01-01"


# ---------------------------------------------------------------------------
# DistributionConfig
# ---------------------------------------------------------------------------


def test_distribution_config_min_season_default() -> None:
    assert DistributionConfig().min_season == 2018


# ---------------------------------------------------------------------------
# ClientConfig validators
# ---------------------------------------------------------------------------


class TestClientConfig:
    def test_defaults(self) -> None:
        cfg = ClientConfig()
        assert cfg.delay == 6.0
        assert cfg.max_retries == 5
        assert cfg.backoff_base == 2.0
        assert cfg.timeout == 120
        assert cfg.use_custom_headers is True

    def test_negative_delay_raises(self) -> None:
        with pytest.raises(ValidationError, match="must be positive"):
            ClientConfig(delay=-1.0)

    def test_zero_delay_raises(self) -> None:
        with pytest.raises(ValidationError, match="must be positive"):
            ClientConfig(delay=0.0)

    def test_negative_backoff_base_raises(self) -> None:
        with pytest.raises(ValidationError, match="must be positive"):
            ClientConfig(backoff_base=-0.5)

    def test_negative_max_retries_raises(self) -> None:
        with pytest.raises(ValidationError, match="must be >= 0"):
            ClientConfig(max_retries=-1)

    def test_zero_max_retries_allowed(self) -> None:
        cfg = ClientConfig(max_retries=0)
        assert cfg.max_retries == 0

    def test_timeout_zero_raises(self) -> None:
        with pytest.raises(ValidationError, match="must be >= 1"):
            ClientConfig(timeout=0)

    def test_timeout_negative_raises(self) -> None:
        with pytest.raises(ValidationError, match="must be >= 1"):
            ClientConfig(timeout=-10)


# ---------------------------------------------------------------------------
# Matchup
# ---------------------------------------------------------------------------


def test_matchup_fields() -> None:
    m = Matchup(home="Brazil", away="France", group="A")
    assert m.home == "Brazil"
    assert m.away == "France"
    assert m.group == "A"


def test_matchup_group_defaults_to_empty_string() -> None:
    m = Matchup(home="Brazil", away="France")
    assert m.group == ""


# ---------------------------------------------------------------------------
# load_config / load_bracket round-trip
# ---------------------------------------------------------------------------


def test_load_config_round_trip(tmp_path: Path) -> None:
    toml_content = """\
[simulation]
n_simulations = 500
classifier = "svm"

[client]
delay = 3.0
max_retries = 2
"""
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(toml_content)

    cfg = load_config(cfg_path)
    assert cfg.simulation.n_simulations == 500
    assert cfg.simulation.classifier == "svm"
    assert cfg.client.delay == 3.0
    assert cfg.client.max_retries == 2


def test_load_config_defaults_for_missing_sections(tmp_path: Path) -> None:
    cfg_path = tmp_path / "empty.toml"
    cfg_path.write_text("")
    cfg = load_config(cfg_path)
    assert isinstance(cfg, AppConfig)
    assert cfg.simulation.n_simulations == 10000


def test_load_bracket_round_trip(tmp_path: Path) -> None:
    toml_content = """\
name = "FIFA World Cup 2026 Bracket"

[[matchups]]
home = "Brazil"
away = "France"
group = "A"

[[matchups]]
home = "Germany"
away = "Argentina"
group = "B"
"""
    bracket_path = tmp_path / "bracket.toml"
    bracket_path.write_text(toml_content)

    bracket = load_bracket(bracket_path)
    assert bracket.name == "FIFA World Cup 2026 Bracket"
    assert len(bracket.matchups) == 2
    assert bracket.matchups[0].home == "Brazil"
    assert bracket.matchups[0].away == "France"
    assert bracket.matchups[0].group == "A"
    assert bracket.matchups[1].home == "Germany"


def test_load_bracket_empty_matchups(tmp_path: Path) -> None:
    toml_content = 'name = "Empty"\n'
    bracket_path = tmp_path / "empty_bracket.toml"
    bracket_path.write_text(toml_content)

    bracket = load_bracket(bracket_path)
    assert bracket.matchups == []
