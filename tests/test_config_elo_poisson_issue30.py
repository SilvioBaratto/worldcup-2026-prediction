"""Source-blind example tests for issue #30: EloConfig + PoissonConfig Pydantic models.

Tests are authored from the acceptance criteria only — no implementation was read.
Each test pins one observable behaviour the implementation must satisfy.
"""

from __future__ import annotations

import pathlib
import tempfile

import pytest
from hypothesis import given, strategies as st

from worldcup_playoff.config import AppConfig, EloConfig, PoissonConfig


# ─────────────────────────────────────────────────────────────────────────────
# EloConfig — required fields and sensible football defaults
# ─────────────────────────────────────────────────────────────────────────────


class TestEloConfigDefaults:
    def test_when_elo_config_is_default_then_initial_rating_is_positive(self):
        assert EloConfig().initial_rating > 0

    def test_when_elo_config_is_default_then_home_advantage_is_non_negative(self):
        assert EloConfig().home_advantage >= 0

    def test_when_elo_config_is_default_then_k_friendly_is_positive(self):
        assert EloConfig().k_friendly > 0

    def test_when_elo_config_is_default_then_k_qualifier_is_positive(self):
        assert EloConfig().k_qualifier > 0

    def test_when_elo_config_is_default_then_k_continental_is_positive(self):
        assert EloConfig().k_continental > 0

    def test_when_elo_config_is_default_then_k_world_cup_is_positive(self):
        assert EloConfig().k_world_cup > 0

    def test_when_elo_config_is_default_then_k_factors_are_sensibly_ordered(self):
        """Football convention: friendly < qualifier ≤ continental ≤ world_cup."""
        cfg = EloConfig()
        assert cfg.k_friendly < cfg.k_qualifier
        assert cfg.k_qualifier <= cfg.k_continental
        assert cfg.k_continental <= cfg.k_world_cup

    def test_when_elo_config_is_default_then_qualifier_keywords_is_non_empty_list(self):
        kws = EloConfig().qualifier_keywords
        assert isinstance(kws, list) and len(kws) > 0

    def test_when_elo_config_is_default_then_continental_keywords_is_non_empty_list(self):
        kws = EloConfig().continental_keywords
        assert isinstance(kws, list) and len(kws) > 0

    def test_when_elo_config_is_default_then_world_cup_keywords_is_non_empty_list(self):
        kws = EloConfig().world_cup_keywords
        assert isinstance(kws, list) and len(kws) > 0

    def test_when_elo_config_is_default_then_all_keyword_lists_contain_strings(self):
        cfg = EloConfig()
        for kw in cfg.qualifier_keywords + cfg.continental_keywords + cfg.world_cup_keywords:
            assert isinstance(kw, str) and len(kw) > 0


class TestEloConfigConstruction:
    def test_when_elo_config_custom_values_are_set_then_all_fields_are_stored(self):
        cfg = EloConfig(
            initial_rating=1000.0,
            home_advantage=50.0,
            k_friendly=10,
            k_qualifier=20,
            k_continental=30,
            k_world_cup=40,
            qualifier_keywords=["qual"],
            continental_keywords=["continental"],
            world_cup_keywords=["worldcup"],
        )
        assert cfg.initial_rating == 1000.0
        assert cfg.home_advantage == 50.0
        assert cfg.k_friendly == 10
        assert cfg.k_qualifier == 20
        assert cfg.k_continental == 30
        assert cfg.k_world_cup == 40
        assert cfg.qualifier_keywords == ["qual"]
        assert cfg.continental_keywords == ["continental"]
        assert cfg.world_cup_keywords == ["worldcup"]


# ─────────────────────────────────────────────────────────────────────────────
# PoissonConfig — required fields and rho_init ≤ 0 constraint
# ─────────────────────────────────────────────────────────────────────────────


class TestPoissonConfigDefaults:
    def test_when_poisson_config_is_default_then_half_life_days_is_positive(self):
        assert PoissonConfig().half_life_days > 0

    def test_when_poisson_config_is_default_then_home_adv_init_attribute_exists(self):
        assert hasattr(PoissonConfig(), "home_adv_init")

    def test_when_poisson_config_is_default_then_rho_init_is_at_most_zero(self):
        assert PoissonConfig().rho_init <= 0

    def test_when_poisson_config_is_default_then_optimizer_maxiter_is_positive(self):
        assert PoissonConfig().optimizer_maxiter > 0

    def test_when_poisson_config_is_default_then_max_goals_is_at_least_one(self):
        assert PoissonConfig().max_goals >= 1

    def test_when_poisson_config_is_default_then_random_seed_attribute_exists(self):
        assert hasattr(PoissonConfig(), "random_seed")


class TestPoissonConfigRhoConstraint:
    def test_when_rho_init_is_zero_then_poisson_config_is_valid(self):
        """rho = 0 is the boundary value; must be accepted."""
        cfg = PoissonConfig(rho_init=0.0)
        assert cfg.rho_init == 0.0

    def test_when_rho_init_is_negative_then_poisson_config_is_valid(self):
        cfg = PoissonConfig(rho_init=-0.5)
        assert cfg.rho_init == -0.5

    def test_when_rho_init_is_positive_then_validation_error_is_raised(self):
        """rho_init is bounded ≤ 0; positive values must be rejected at construction."""
        with pytest.raises(Exception):
            PoissonConfig(rho_init=0.1)


# Property: the rho_init ≤ 0 bound must hold for ALL positive floats.
@given(st.floats(min_value=1e-9, max_value=1e9, allow_nan=False, allow_infinity=False))
def test_when_rho_init_is_any_strictly_positive_value_then_poisson_config_raises(pos_rho):
    """Invariant: no positive rho_init is ever accepted."""
    with pytest.raises(Exception):
        PoissonConfig(rho_init=pos_rho)


# ─────────────────────────────────────────────────────────────────────────────
# AppConfig — elo and poisson fields exposed
# ─────────────────────────────────────────────────────────────────────────────


class TestAppConfigExposesEloAndPoisson:
    def test_when_app_config_is_default_then_elo_attribute_is_elo_config_instance(self):
        assert isinstance(AppConfig().elo, EloConfig)

    def test_when_app_config_is_default_then_poisson_attribute_is_poisson_config_instance(self):
        assert isinstance(AppConfig().poisson, PoissonConfig)

    def test_when_app_config_elo_is_inspected_then_initial_rating_default_matches(self):
        """AppConfig.elo must carry the same defaults as a standalone EloConfig()."""
        assert AppConfig().elo.initial_rating == EloConfig().initial_rating

    def test_when_app_config_poisson_is_inspected_then_rho_init_is_at_most_zero(self):
        """AppConfig.poisson.rho_init must satisfy the ≤ 0 bound via defaults."""
        assert AppConfig().poisson.rho_init <= 0


# ─────────────────────────────────────────────────────────────────────────────
# TOML wiring — [elo] and [poisson] sections parsed into AppConfig
# ─────────────────────────────────────────────────────────────────────────────


def _write_toml(text: str) -> pathlib.Path:
    f = tempfile.NamedTemporaryFile(suffix=".toml", mode="w", delete=False, encoding="utf-8")
    f.write(text)
    f.close()
    return pathlib.Path(f.name)


class TestAppConfigTomlEloSection:
    def test_when_toml_elo_section_sets_initial_rating_then_app_config_reflects_it(self):
        from worldcup_playoff.config import load_config

        path = _write_toml("[elo]\ninitial_rating = 1600.0\n")
        cfg = load_config(path)
        assert cfg.elo.initial_rating == 1600.0

    def test_when_toml_elo_section_sets_home_advantage_then_app_config_reflects_it(self):
        from worldcup_playoff.config import load_config

        path = _write_toml("[elo]\nhome_advantage = 80.0\n")
        cfg = load_config(path)
        assert cfg.elo.home_advantage == 80.0

    def test_when_toml_elo_section_sets_k_friendly_then_app_config_reflects_it(self):
        from worldcup_playoff.config import load_config

        path = _write_toml("[elo]\nk_friendly = 15\n")
        cfg = load_config(path)
        assert cfg.elo.k_friendly == 15

    def test_when_toml_elo_section_sets_k_world_cup_then_app_config_reflects_it(self):
        from worldcup_playoff.config import load_config

        path = _write_toml("[elo]\nk_world_cup = 55\n")
        cfg = load_config(path)
        assert cfg.elo.k_world_cup == 55

    def test_when_toml_elo_section_sets_qualifier_keywords_then_app_config_reflects_it(self):
        from worldcup_playoff.config import load_config

        path = _write_toml('[elo]\nqualifier_keywords = ["qual", "qualifying"]\n')
        cfg = load_config(path)
        assert "qual" in cfg.elo.qualifier_keywords

    def test_when_toml_has_no_elo_section_then_elo_defaults_are_used(self):
        from worldcup_playoff.config import load_config

        path = _write_toml("")
        cfg = load_config(path)
        assert isinstance(cfg.elo, EloConfig)
        assert cfg.elo.initial_rating > 0


class TestAppConfigTomlPoissonSection:
    def test_when_toml_poisson_section_sets_half_life_days_then_app_config_reflects_it(self):
        from worldcup_playoff.config import load_config

        path = _write_toml("[poisson]\nhalf_life_days = 730.0\n")
        cfg = load_config(path)
        assert cfg.poisson.half_life_days == 730.0

    def test_when_toml_poisson_section_sets_rho_init_then_app_config_reflects_it(self):
        from worldcup_playoff.config import load_config

        path = _write_toml("[poisson]\nrho_init = -0.2\n")
        cfg = load_config(path)
        assert cfg.poisson.rho_init == -0.2

    def test_when_toml_poisson_section_sets_max_goals_then_app_config_reflects_it(self):
        from worldcup_playoff.config import load_config

        path = _write_toml("[poisson]\nmax_goals = 8\n")
        cfg = load_config(path)
        assert cfg.poisson.max_goals == 8

    def test_when_toml_poisson_section_sets_optimizer_maxiter_then_app_config_reflects_it(self):
        from worldcup_playoff.config import load_config

        path = _write_toml("[poisson]\noptimizer_maxiter = 500\n")
        cfg = load_config(path)
        assert cfg.poisson.optimizer_maxiter == 500

    def test_when_toml_poisson_section_sets_random_seed_then_app_config_reflects_it(self):
        from worldcup_playoff.config import load_config

        path = _write_toml("[poisson]\nrandom_seed = 99\n")
        cfg = load_config(path)
        assert cfg.poisson.random_seed == 99

    def test_when_toml_has_no_poisson_section_then_poisson_defaults_are_used(self):
        from worldcup_playoff.config import load_config

        path = _write_toml("")
        cfg = load_config(path)
        assert isinstance(cfg.poisson, PoissonConfig)
        assert cfg.poisson.half_life_days > 0


class TestDefaultTomlWiring:
    """Tests that config/default.toml contains [elo] and [poisson] sections."""

    _DEFAULT_TOML = pathlib.Path("config/default.toml")

    def test_when_default_toml_is_loaded_then_elo_field_is_elo_config(self):
        from worldcup_playoff.config import load_config

        if not self._DEFAULT_TOML.exists():
            pytest.skip("config/default.toml not present")
        cfg = load_config(self._DEFAULT_TOML)
        assert isinstance(cfg.elo, EloConfig)

    def test_when_default_toml_is_loaded_then_poisson_field_is_poisson_config(self):
        from worldcup_playoff.config import load_config

        if not self._DEFAULT_TOML.exists():
            pytest.skip("config/default.toml not present")
        cfg = load_config(self._DEFAULT_TOML)
        assert isinstance(cfg.poisson, PoissonConfig)

    def test_when_default_toml_is_loaded_then_elo_initial_rating_is_positive(self):
        from worldcup_playoff.config import load_config

        if not self._DEFAULT_TOML.exists():
            pytest.skip("config/default.toml not present")
        cfg = load_config(self._DEFAULT_TOML)
        assert cfg.elo.initial_rating > 0

    def test_when_default_toml_is_loaded_then_poisson_rho_init_is_at_most_zero(self):
        from worldcup_playoff.config import load_config

        if not self._DEFAULT_TOML.exists():
            pytest.skip("config/default.toml not present")
        cfg = load_config(self._DEFAULT_TOML)
        assert cfg.poisson.rho_init <= 0
