"""Source-blind example tests for issue #43: Cycle 5 config verification + OddsConfig extension.

Derived exclusively from the acceptance criteria. No implementation source was read.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Criterion: load_config(config/default.toml) round-trips every Cycle-5 section
#            without error.
# ---------------------------------------------------------------------------


def test_when_default_toml_is_loaded_then_no_error_is_raised():
    """load_config must succeed on the real config/default.toml without raising."""
    from worldcup_playoff.config import load_config

    config_path = Path("config/default.toml")
    cfg = load_config(config_path)
    assert cfg is not None


def test_when_default_toml_is_loaded_then_poisson_section_is_present():
    """[poisson] section must be accessible as AppConfig.poisson after load."""
    from worldcup_playoff.config import load_config

    cfg = load_config(Path("config/default.toml"))
    assert cfg.poisson is not None


def test_when_default_toml_is_loaded_then_elo_section_is_present():
    """[elo] section must be accessible as AppConfig.elo after load."""
    from worldcup_playoff.config import load_config

    cfg = load_config(Path("config/default.toml"))
    assert cfg.elo is not None


def test_when_default_toml_is_loaded_then_rf_section_is_present():
    """[rf] section must be accessible as AppConfig.rf after load."""
    from worldcup_playoff.config import load_config

    cfg = load_config(Path("config/default.toml"))
    assert cfg.rf is not None


def test_when_default_toml_is_loaded_then_odds_section_is_present():
    """[odds] section must be accessible as AppConfig.odds after load."""
    from worldcup_playoff.config import load_config

    cfg = load_config(Path("config/default.toml"))
    assert cfg.odds is not None


def test_when_default_toml_is_loaded_then_live_section_is_present():
    """[live] section must be accessible as AppConfig.live after load."""
    from worldcup_playoff.config import load_config

    cfg = load_config(Path("config/default.toml"))
    assert cfg.live is not None


# ---------------------------------------------------------------------------
# Criterion: OddsConfig gains `markets: list[str] = ["outright", "match"]`
#            and a `match_url_template: str` field.
# ---------------------------------------------------------------------------


def test_when_odds_config_is_default_constructed_then_markets_contains_outright_and_match():
    """OddsConfig() must have markets == ['outright', 'match'] by default."""
    from worldcup_playoff.config import OddsConfig

    cfg = OddsConfig()
    assert cfg.markets == ["outright", "match"]


def test_when_odds_config_is_default_constructed_then_match_url_template_field_exists():
    """OddsConfig() must expose a match_url_template attribute."""
    from worldcup_playoff.config import OddsConfig

    cfg = OddsConfig()
    assert hasattr(cfg, "match_url_template")


def test_when_odds_config_is_default_constructed_then_match_url_template_is_a_string():
    """match_url_template must be a str (not None, not int)."""
    from worldcup_playoff.config import OddsConfig

    cfg = OddsConfig()
    assert isinstance(cfg.match_url_template, str)


def test_when_odds_config_markets_is_set_then_value_is_stored():
    """markets accepts a custom non-empty list and stores it."""
    from worldcup_playoff.config import OddsConfig

    cfg = OddsConfig(markets=["match"])
    assert cfg.markets == ["match"]


def test_when_odds_config_match_url_template_is_set_then_value_is_stored():
    """match_url_template accepts a custom string and stores it."""
    from worldcup_playoff.config import OddsConfig

    url = "https://example.com/{year}/match/{home_team}/{away_team}"
    cfg = OddsConfig(match_url_template=url)
    assert cfg.match_url_template == url


# ---------------------------------------------------------------------------
# Criterion: field validator — markets must be non-empty (mirrors existing style).
# ---------------------------------------------------------------------------


def test_when_odds_config_markets_is_empty_then_validation_error_is_raised():
    """OddsConfig with markets=[] must raise a ValidationError."""
    from pydantic import ValidationError

    from worldcup_playoff.config import OddsConfig

    with pytest.raises(ValidationError):
        OddsConfig(markets=[])


# Property: for any non-empty list of strings, OddsConfig.markets validation passes.
@given(markets=st.lists(st.text(min_size=1), min_size=1, max_size=10))
@settings(max_examples=50)
def test_when_markets_is_non_empty_then_odds_config_validates_without_error(
    markets: list[str],
) -> None:
    """Any non-empty list[str] must be accepted by the markets validator."""
    from worldcup_playoff.config import OddsConfig

    cfg = OddsConfig(markets=markets)
    assert cfg.markets == markets


# ---------------------------------------------------------------------------
# Criterion: New OddsConfig fields default-load when absent from TOML
#            (extra="ignore" preserved).
# ---------------------------------------------------------------------------


def test_when_odds_config_is_constructed_without_markets_keyword_then_defaults_apply():
    """Constructing OddsConfig() with no arguments must not raise."""
    from worldcup_playoff.config import OddsConfig

    cfg = OddsConfig()
    assert cfg.markets is not None
    assert cfg.match_url_template is not None


def test_when_toml_odds_section_omits_markets_then_app_config_loads_without_error(
    tmp_path: Path,
) -> None:
    """A TOML that has [odds] but no markets/match_url_template must still load."""
    from worldcup_playoff.config import load_config

    toml_content = """\
[odds]
seasons = [2014, 2018, 2022]
"""
    toml_file = tmp_path / "test_config.toml"
    toml_file.write_text(toml_content, encoding="utf-8")

    cfg = load_config(toml_file)
    assert cfg.odds.markets == ["outright", "match"]


def test_when_toml_has_no_odds_section_then_app_config_still_loads(tmp_path: Path) -> None:
    """A TOML with no [odds] section must load and expose OddsConfig defaults."""
    from worldcup_playoff.config import load_config

    toml_content = """\
[simulation]
n_simulations = 100
"""
    toml_file = tmp_path / "minimal.toml"
    toml_file.write_text(toml_content, encoding="utf-8")

    cfg = load_config(toml_file)
    # New fields must be available at their defaults even when absent from TOML.
    assert cfg.odds.markets == ["outright", "match"]
    assert isinstance(cfg.odds.match_url_template, str)


def test_when_toml_odds_section_has_unknown_keys_then_they_are_ignored(tmp_path: Path) -> None:
    """extra='ignore' must still hold — unknown keys in [odds] must not raise."""
    from worldcup_playoff.config import load_config

    toml_content = """\
[odds]
seasons = [2022]
unknown_future_key = "value"
"""
    toml_file = tmp_path / "extra_keys.toml"
    toml_file.write_text(toml_content, encoding="utf-8")

    cfg = load_config(toml_file)
    assert cfg.odds is not None


# ---------------------------------------------------------------------------
# Criterion: AppConfig with defaults constructs OddsConfig as its odds attribute.
# ---------------------------------------------------------------------------


def test_when_app_config_is_constructed_with_defaults_then_odds_attribute_is_odds_config():
    """AppConfig() must expose .odds as an OddsConfig instance."""
    from worldcup_playoff.config import AppConfig, OddsConfig

    cfg = AppConfig()
    assert isinstance(cfg.odds, OddsConfig)


def test_when_app_config_is_constructed_with_defaults_then_odds_has_markets_field():
    """AppConfig().odds.markets must be a non-empty list."""
    from worldcup_playoff.config import AppConfig

    cfg = AppConfig()
    assert isinstance(cfg.odds.markets, list)
    assert len(cfg.odds.markets) > 0


def test_when_app_config_is_constructed_with_defaults_then_odds_has_match_url_template():
    """AppConfig().odds.match_url_template must be a str."""
    from worldcup_playoff.config import AppConfig

    cfg = AppConfig()
    assert isinstance(cfg.odds.match_url_template, str)
