"""Source-blind example tests for issue #16: Cycle 5 config groundwork.

Tests authored from acceptance criteria only, before any implementation exists.
Issue: feat: Cycle 5 config groundwork — [rf]/[odds] sections, knockout sim params, TOML wiring
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given, strategies as st
from pydantic import ValidationError

from worldcup_playoff.config import (
    AppConfig,
    OddsConfig,
    RFConfig,
    SimulationConfig,
    load_config,
)

_DEFAULT_TOML = Path("config/default.toml")


# ---------------------------------------------------------------------------
# OddsConfig — fields and validators
# ---------------------------------------------------------------------------


def test_when_odds_config_created_with_defaults_then_seasons_are_2014_2018_2022():
    cfg = OddsConfig()
    assert cfg.seasons == [2014, 2018, 2022]


def test_when_odds_config_created_with_defaults_then_request_timeout_is_positive():
    cfg = OddsConfig()
    assert cfg.request_timeout > 0


def test_when_odds_config_created_with_defaults_then_cache_dir_is_present():
    cfg = OddsConfig()
    assert cfg.cache_dir is not None


def test_when_odds_config_created_with_defaults_then_user_agent_is_a_string():
    cfg = OddsConfig()
    assert isinstance(cfg.user_agent, str)


def test_when_odds_config_created_with_defaults_then_enabled_is_a_bool():
    cfg = OddsConfig()
    assert isinstance(cfg.enabled, bool)


def test_when_odds_config_timeout_is_zero_then_validation_error_is_raised():
    with pytest.raises(ValidationError):
        OddsConfig(request_timeout=0)


def test_when_odds_config_timeout_is_negative_then_validation_error_is_raised():
    with pytest.raises(ValidationError):
        OddsConfig(request_timeout=-5)


def test_when_odds_config_seasons_is_empty_then_validation_error_is_raised():
    with pytest.raises(ValidationError):
        OddsConfig(seasons=[])


@given(st.lists(st.integers(min_value=1900, max_value=2100), min_size=1))
def test_when_odds_config_seasons_is_non_empty_then_no_validation_error_is_raised(seasons):
    """Invariant: any non-empty list of plausible years is accepted by OddsConfig."""
    OddsConfig(seasons=seasons)  # must not raise


# ---------------------------------------------------------------------------
# RFConfig — hybrid RF/GBM tuning surface
# ---------------------------------------------------------------------------


def test_when_rf_config_created_with_defaults_then_instance_is_returned():
    cfg = RFConfig()
    assert cfg is not None


# ---------------------------------------------------------------------------
# SimulationConfig — extended knockout fields (must not break existing fields)
# ---------------------------------------------------------------------------


def test_when_simulation_config_created_with_defaults_then_extra_time_factor_is_0_33():
    cfg = SimulationConfig()
    assert cfg.extra_time_factor == pytest.approx(0.33)


def test_when_simulation_config_created_with_defaults_then_random_seed_is_an_int():
    cfg = SimulationConfig()
    assert isinstance(cfg.random_seed, int)


def test_when_simulation_config_random_seed_is_zero_then_config_is_accepted():
    """Zero is a valid numpy seed; per reviewer guidance random_seed validates >= 0."""
    cfg = SimulationConfig(random_seed=0)
    assert cfg.random_seed == 0


def test_when_simulation_config_random_seed_is_negative_then_validation_error_is_raised():
    with pytest.raises(ValidationError):
        SimulationConfig(random_seed=-1)


def test_when_simulation_config_created_with_defaults_then_n_simulations_is_still_present():
    """Extending SimulationConfig must not remove the existing n_simulations field."""
    cfg = SimulationConfig()
    assert cfg.n_simulations >= 1


def test_when_simulation_config_created_with_defaults_then_classifier_is_still_present():
    """Extending SimulationConfig must not remove the existing classifier field."""
    cfg = SimulationConfig()
    assert isinstance(cfg.classifier, str) and cfg.classifier


def test_when_simulation_config_n_simulations_is_positive_then_config_is_accepted():
    cfg = SimulationConfig(n_simulations=500)
    assert cfg.n_simulations == 500


def test_when_simulation_config_n_simulations_is_zero_then_validation_error_is_raised():
    """Existing n_simulations validator must still reject zero after extension."""
    with pytest.raises(ValidationError):
        SimulationConfig(n_simulations=0)


# ---------------------------------------------------------------------------
# AppConfig — both new models wired in as attributes
# ---------------------------------------------------------------------------


def test_when_app_config_created_with_defaults_then_odds_attribute_is_odds_config():
    cfg = AppConfig()
    assert isinstance(cfg.odds, OddsConfig)


def test_when_app_config_created_with_defaults_then_rf_attribute_is_rf_config():
    cfg = AppConfig()
    assert isinstance(cfg.rf, RFConfig)


# ---------------------------------------------------------------------------
# load_config — default.toml round-trips and new-section exposure
# ---------------------------------------------------------------------------


def test_when_default_toml_is_loaded_then_odds_config_is_returned():
    cfg = load_config(_DEFAULT_TOML)
    assert isinstance(cfg.odds, OddsConfig)


def test_when_default_toml_is_loaded_then_rf_config_is_returned():
    cfg = load_config(_DEFAULT_TOML)
    assert isinstance(cfg.rf, RFConfig)


def test_when_default_toml_is_loaded_then_simulation_extra_time_factor_is_accessible():
    cfg = load_config(_DEFAULT_TOML)
    assert hasattr(cfg.simulation, "extra_time_factor")


def test_when_default_toml_is_loaded_then_simulation_random_seed_is_accessible():
    cfg = load_config(_DEFAULT_TOML)
    assert hasattr(cfg.simulation, "random_seed")


def test_when_default_toml_is_loaded_then_live_section_is_accessible():
    cfg = load_config(_DEFAULT_TOML)
    assert hasattr(cfg, "live")


def test_when_default_toml_is_loaded_then_elo_section_is_accessible():
    cfg = load_config(_DEFAULT_TOML)
    assert hasattr(cfg, "elo")


def test_when_default_toml_is_loaded_then_poisson_section_is_accessible():
    cfg = load_config(_DEFAULT_TOML)
    assert hasattr(cfg, "poisson")


def test_when_default_toml_is_loaded_then_odds_seasons_match_expected_defaults():
    """Default odds.seasons from TOML must round-trip to [2014, 2018, 2022]."""
    cfg = load_config(_DEFAULT_TOML)
    assert cfg.odds.seasons == [2014, 2018, 2022]


def test_when_default_toml_is_loaded_then_odds_request_timeout_is_positive():
    cfg = load_config(_DEFAULT_TOML)
    assert cfg.odds.request_timeout > 0


def test_when_default_toml_is_loaded_then_simulation_extra_time_factor_is_0_33():
    cfg = load_config(_DEFAULT_TOML)
    assert cfg.simulation.extra_time_factor == pytest.approx(0.33)


def test_when_toml_has_negative_odds_timeout_then_load_config_raises_validation_error(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_bytes(b"[odds]\nrequest_timeout = -1\n")
    with pytest.raises(ValidationError):
        load_config(bad)


def test_when_toml_has_empty_odds_seasons_then_load_config_raises_validation_error(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_bytes(b"[odds]\nseasons = []\n")
    with pytest.raises(ValidationError):
        load_config(bad)


def test_when_toml_has_negative_simulation_random_seed_then_load_config_raises_validation_error(
    tmp_path,
):
    bad = tmp_path / "bad.toml"
    bad.write_bytes(b"[simulation]\nrandom_seed = -1\n")
    with pytest.raises(ValidationError):
        load_config(bad)


def test_when_default_toml_is_loaded_twice_then_results_are_identical():
    """Round-trip invariant: loading the same file twice must yield identical config."""
    a = load_config(_DEFAULT_TOML)
    b = load_config(_DEFAULT_TOML)
    assert a.model_dump() == b.model_dump()
