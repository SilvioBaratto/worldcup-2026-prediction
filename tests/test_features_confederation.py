"""Source-blind example tests for issue #7: confederation map + ranking resolver + config.

All tests are derived solely from the acceptance criteria. No implementation source
was read during authorship (Red-phase TDD).
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings, strategies as st
from pydantic import ValidationError

from worldcup_playoff.features.confederation import (
    CONFEDERATION_MAP,
    CONFEDERATIONS,
    RankingResolution,
    confederation_of,
    resolve_ranking,
)
from worldcup_playoff.config import AppConfig, FeatureBuildConfig
from worldcup_playoff.data.crosswalk import normalize_team

# ---------------------------------------------------------------------------
# Criterion 1 — explicit spot-checks for known WC2026 teams + alias resolution
# ---------------------------------------------------------------------------


def test_when_brazil_is_queried_then_conmebol_is_returned():
    assert confederation_of("Brazil") == "CONMEBOL"


def test_when_united_states_is_queried_then_concacaf_is_returned():
    assert confederation_of("United States") == "CONCACAF"


def test_when_turkiye_alias_is_queried_then_uefa_is_returned():
    # "Türkiye" is a crosswalk alias that normalizes to "Turkey" → UEFA
    assert confederation_of("Türkiye") == "UEFA"


def test_when_japan_is_queried_then_afc_is_returned():
    assert confederation_of("Japan") == "AFC"


def test_when_new_zealand_is_queried_then_ofc_is_returned():
    assert confederation_of("New Zealand") == "OFC"


# ---------------------------------------------------------------------------
# Criterion 2 — parametrised WC2026 teams + unknown-team guard
# ---------------------------------------------------------------------------

# Representative subset of WC2026 qualified nations; each must map to exactly
# one confederation without raising.  The parametrize list deliberately spans
# all six confederations so the mapping is exercised end-to-end.
_WC2026_SAMPLE: list[tuple[str, str]] = [
    # CONMEBOL
    ("Argentina", "CONMEBOL"),
    ("Brazil", "CONMEBOL"),
    ("Uruguay", "CONMEBOL"),
    ("Ecuador", "CONMEBOL"),
    ("Colombia", "CONMEBOL"),
    # UEFA
    ("France", "UEFA"),
    ("England", "UEFA"),
    ("Germany", "UEFA"),
    ("Spain", "UEFA"),
    ("Portugal", "UEFA"),
    ("Netherlands", "UEFA"),
    ("Belgium", "UEFA"),
    ("Turkey", "UEFA"),
    ("Croatia", "UEFA"),
    ("Serbia", "UEFA"),
    ("Switzerland", "UEFA"),
    ("Austria", "UEFA"),
    ("Poland", "UEFA"),
    ("Albania", "UEFA"),
    # CONCACAF
    ("United States", "CONCACAF"),
    ("Canada", "CONCACAF"),
    ("Mexico", "CONCACAF"),
    ("Jamaica", "CONCACAF"),
    ("Panama", "CONCACAF"),
    ("Honduras", "CONCACAF"),
    # AFC
    ("Japan", "AFC"),
    ("South Korea", "AFC"),
    ("Iran", "AFC"),
    ("Saudi Arabia", "AFC"),
    ("Australia", "AFC"),
    ("Iraq", "AFC"),
    ("Jordan", "AFC"),
    ("Uzbekistan", "AFC"),
    # CAF
    ("Morocco", "CAF"),
    ("Nigeria", "CAF"),
    ("Senegal", "CAF"),
    ("Egypt", "CAF"),
    ("Ivory Coast", "CAF"),
    ("South Africa", "CAF"),
    ("Algeria", "CAF"),
    ("Tunisia", "CAF"),
    ("Cameroon", "CAF"),
    ("DR Congo", "CAF"),
    # OFC
    ("New Zealand", "OFC"),
]


@pytest.mark.parametrize("team,expected", _WC2026_SAMPLE)
def test_when_wc2026_team_is_queried_then_correct_confederation_is_returned(
    team: str, expected: str
) -> None:
    assert confederation_of(team) == expected, (
        f"Expected {team!r} → {expected!r}, got {confederation_of(team)!r}"
    )


def test_when_unknown_team_is_queried_then_none_is_returned():
    assert confederation_of("Atlantis FC") is None


def test_when_unknown_team_is_queried_then_no_exception_is_raised():
    # Must degrade gracefully — never raise for an unrecognised name
    result = confederation_of("Planet Football United")
    assert result is None


# ---------------------------------------------------------------------------
# Criterion 3 — CONFEDERATIONS tuple structure; CONFEDERATION_MAP key canonicity
# ---------------------------------------------------------------------------


def test_when_confederations_is_inspected_then_it_is_a_tuple():
    assert isinstance(CONFEDERATIONS, tuple)


def test_when_confederations_is_inspected_then_it_has_six_members():
    assert len(CONFEDERATIONS) == 6


def test_when_confederations_is_inspected_then_it_contains_all_six_bodies():
    assert set(CONFEDERATIONS) == {"UEFA", "CONMEBOL", "CAF", "AFC", "CONCACAF", "OFC"}


def test_when_confederation_map_keys_are_normalized_then_round_trip_holds():
    """Every key in CONFEDERATION_MAP must already be canonical.

    A non-canonical key (e.g. 'Türkiye' instead of 'Turkey') would silently
    miss lookups when callers pre-normalize team names through the crosswalk.
    """
    non_canonical = [k for k in CONFEDERATION_MAP if normalize_team(k) != k]
    assert non_canonical == [], f"Non-canonical keys found in CONFEDERATION_MAP: {non_canonical}"


# ---------------------------------------------------------------------------
# Criterion 4 — resolve_ranking with ranking=None → used_fallback=True
# ---------------------------------------------------------------------------


def test_when_ranking_is_none_then_resolution_has_used_fallback_true():
    result = resolve_ranking(
        "Brazil",
        ranking=None,
        as_of="2020-12-10",
        staleness_cutoff="2020-12-10",
    )
    assert isinstance(result, RankingResolution)
    assert result.used_fallback is True


def test_when_ranking_is_none_then_resolution_value_is_none():
    result = resolve_ranking(
        "Brazil",
        ranking=None,
        as_of="2020-12-10",
        staleness_cutoff="2020-12-10",
    )
    assert result.value is None


def test_when_ranking_is_none_then_confederation_is_populated():
    result = resolve_ranking(
        "Brazil",
        ranking=None,
        as_of="2020-12-10",
        staleness_cutoff="2020-12-10",
    )
    assert result.confederation == "CONMEBOL"


def test_when_ranking_is_none_and_team_is_afc_then_confederation_is_afc():
    result = resolve_ranking(
        "Japan",
        ranking=None,
        as_of="2020-12-10",
        staleness_cutoff="2020-12-10",
    )
    assert result.confederation == "AFC"


# ---------------------------------------------------------------------------
# Criterion 5 — fresh ranking, stale ranking, absent team
# ---------------------------------------------------------------------------


def test_when_ranking_is_fresh_and_team_present_then_used_fallback_is_false():
    result = resolve_ranking(
        "Brazil",
        ranking={"Brazil": 1750.5},
        as_of="2020-12-10",
        staleness_cutoff="2020-12-10",
    )
    assert result.used_fallback is False


def test_when_ranking_is_fresh_and_team_present_then_value_is_populated():
    result = resolve_ranking(
        "Brazil",
        ranking={"Brazil": 1750.5},
        as_of="2020-12-10",
        staleness_cutoff="2020-12-10",
    )
    assert result.value == 1750.5


def test_when_ranking_is_fresh_then_confederation_is_still_populated():
    result = resolve_ranking(
        "Brazil",
        ranking={"Brazil": 1750.5},
        as_of="2020-12-10",
        staleness_cutoff="2020-12-10",
    )
    assert result.confederation == "CONMEBOL"


def test_when_as_of_is_after_staleness_cutoff_then_fallback_path_is_taken():
    result = resolve_ranking(
        "Brazil",
        ranking={"Brazil": 1750.5},
        as_of="2021-01-01",
        staleness_cutoff="2020-12-10",
    )
    assert result.used_fallback is True
    assert result.value is None


def test_when_as_of_is_before_staleness_cutoff_then_fallback_is_not_taken():
    result = resolve_ranking(
        "France",
        ranking={"France": 1850.0},
        as_of="2020-06-01",
        staleness_cutoff="2020-12-10",
    )
    assert result.used_fallback is False
    assert result.value == 1850.0


def test_when_team_is_absent_from_ranking_then_fallback_path_is_taken():
    result = resolve_ranking(
        "Japan",
        ranking={"France": 1800.0},
        as_of="2020-06-01",
        staleness_cutoff="2020-12-10",
    )
    assert result.used_fallback is True
    assert result.value is None


def test_when_team_is_absent_from_ranking_then_confederation_is_populated():
    result = resolve_ranking(
        "Japan",
        ranking={"France": 1800.0},
        as_of="2020-06-01",
        staleness_cutoff="2020-12-10",
    )
    assert result.confederation == "AFC"


# ---------------------------------------------------------------------------
# Criterion 6 — FeatureBuildConfig defaults and validation; AppConfig wiring
# ---------------------------------------------------------------------------


def test_when_feature_build_config_is_default_then_staleness_cutoff_is_correct():
    assert FeatureBuildConfig().ranking_staleness_cutoff == "2020-12-10"


def test_when_feature_build_config_is_default_then_form_window_is_five():
    assert FeatureBuildConfig().form_window == 5


def test_when_feature_build_config_is_default_then_half_life_is_365():
    assert FeatureBuildConfig().form_half_life_days == 365.0


def test_when_feature_build_config_is_default_then_random_seed_is_42():
    assert FeatureBuildConfig().random_seed == 42


def test_when_feature_build_config_is_default_then_confederation_fallback_is_true():
    assert FeatureBuildConfig().confederation_fallback is True


def test_when_form_window_is_zero_then_validation_error_is_raised():
    with pytest.raises(ValidationError):
        FeatureBuildConfig(form_window=0)


def test_when_form_window_is_negative_then_validation_error_is_raised():
    with pytest.raises(ValidationError):
        FeatureBuildConfig(form_window=-3)


def test_when_form_half_life_days_is_zero_then_validation_error_is_raised():
    with pytest.raises(ValidationError):
        FeatureBuildConfig(form_half_life_days=0.0)


def test_when_form_half_life_days_is_negative_then_validation_error_is_raised():
    with pytest.raises(ValidationError):
        FeatureBuildConfig(form_half_life_days=-1.0)


def test_when_app_config_is_default_then_features_build_is_feature_build_config():
    assert isinstance(AppConfig().features_build, FeatureBuildConfig)


def test_when_app_config_features_build_is_inspected_then_defaults_match():
    # Ensures the AppConfig wiring uses the same defaults, not overriding them
    cfg = AppConfig().features_build
    assert cfg.form_window == 5
    assert cfg.form_half_life_days == 365.0


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------


@given(st.text())
@settings(max_examples=200)
def test_when_any_string_is_queried_then_confederation_of_never_raises(name: str) -> None:
    """confederation_of is total over all strings: unknown inputs return None, never raise."""
    result = confederation_of(name)
    assert result is None or result in CONFEDERATIONS


@given(st.sampled_from(sorted(CONFEDERATION_MAP)))
def test_when_confederation_map_key_is_normalised_then_it_is_unchanged(team: str) -> None:
    """Round-trip invariant: every key in CONFEDERATION_MAP is already canonical."""
    assert normalize_team(team) == team
