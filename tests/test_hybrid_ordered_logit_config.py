"""Tests for the HybridConfig model (RF/GBM goal-based hybrid).

Pins down the observable behaviour of the hybrid config surface used by the
backtest/calibration path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from worldcup_playoff.config import AppConfig, HybridConfig, load_config

# Path to the project's default config (one level up from tests/)
_DEFAULT_TOML = Path(__file__).parent.parent / "config" / "default.toml"


# ---------------------------------------------------------------------------
# AppConfig exposes .hybrid
# ---------------------------------------------------------------------------


def test_when_app_config_default_then_hybrid_attribute_exists():
    cfg = AppConfig()
    assert hasattr(cfg, "hybrid")


def test_when_app_config_default_then_hybrid_is_hybrid_config_instance():
    cfg = AppConfig()
    assert isinstance(cfg.hybrid, HybridConfig)


# ---------------------------------------------------------------------------
# HybridConfig documented defaults
# ---------------------------------------------------------------------------


def test_when_hybrid_config_default_then_rf_n_estimators_is_positive():
    assert HybridConfig().rf_n_estimators > 0


def test_when_hybrid_config_default_then_gb_n_estimators_is_positive():
    assert HybridConfig().gb_n_estimators > 0


def test_when_hybrid_config_default_then_gb_learning_rate_is_positive():
    assert HybridConfig().gb_learning_rate > 0.0


def test_when_hybrid_config_default_then_max_goals_is_positive():
    assert HybridConfig().max_goals > 0


def test_when_hybrid_config_default_then_test_size_is_in_open_unit_interval():
    ts = HybridConfig().test_size
    assert 0.0 < ts < 1.0


def test_when_hybrid_config_default_then_random_seed_is_an_int():
    assert isinstance(HybridConfig().random_seed, int)


def test_when_hybrid_config_test_size_is_zero_then_validation_error_is_raised():
    with pytest.raises(Exception):
        HybridConfig(test_size=0.0)


def test_when_hybrid_config_test_size_is_one_then_validation_error_is_raised():
    with pytest.raises(Exception):
        HybridConfig(test_size=1.0)


def test_when_hybrid_config_test_size_is_negative_then_validation_error_is_raised():
    with pytest.raises(Exception):
        HybridConfig(test_size=-0.1)


def test_when_hybrid_config_test_size_is_above_one_then_validation_error_is_raised():
    with pytest.raises(Exception):
        HybridConfig(test_size=1.1)


# ---------------------------------------------------------------------------
# config/default.toml has a [hybrid] table; load_config round-trips overrides
# ---------------------------------------------------------------------------


def test_when_default_toml_exists_then_hybrid_section_is_present():
    if not _DEFAULT_TOML.exists():
        pytest.fail(f"config/default.toml not found at {_DEFAULT_TOML}")
    import tomllib

    with open(_DEFAULT_TOML, "rb") as f:
        raw = tomllib.load(f)
    assert "hybrid" in raw, "config/default.toml must contain a [hybrid] table"


def test_when_load_config_with_hybrid_rf_n_estimators_override_then_value_is_loaded(tmp_path):
    p = tmp_path / "cfg.toml"
    p.write_text("[hybrid]\nrf_n_estimators = 500\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.hybrid.rf_n_estimators == 500


def test_when_load_config_with_hybrid_test_size_override_then_value_is_loaded(tmp_path):
    p = tmp_path / "cfg.toml"
    p.write_text("[hybrid]\ntest_size = 0.15\n", encoding="utf-8")
    cfg = load_config(p)
    assert abs(cfg.hybrid.test_size - 0.15) < 1e-9


def test_when_load_config_with_empty_file_then_hybrid_defaults_are_applied(tmp_path):
    p = tmp_path / "cfg.toml"
    p.write_text("# empty\n", encoding="utf-8")
    cfg = load_config(p)
    assert isinstance(cfg.hybrid, HybridConfig)
    assert 0.0 < cfg.hybrid.test_size < 1.0
