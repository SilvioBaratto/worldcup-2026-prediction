"""Source-blind example tests for issue #34.

Derived from acceptance criteria only — no implementation source was read.
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


# ── Criterion 1 — FeatureBuildConfig defaults ──────────────────────────────


def test_when_feature_build_config_is_created_then_ranking_staleness_cutoff_is_default():
    assert FeatureBuildConfig().ranking_staleness_cutoff == "2020-12-10"


def test_when_feature_build_config_is_created_then_form_window_is_5():
    assert FeatureBuildConfig().form_window == 5


def test_when_feature_build_config_is_created_then_form_half_life_days_is_365():
    assert FeatureBuildConfig().form_half_life_days == 365.0


def test_when_feature_build_config_is_created_then_random_seed_is_42():
    assert FeatureBuildConfig().random_seed == 42


def test_when_feature_build_config_is_created_then_confederation_fallback_is_true():
    assert FeatureBuildConfig().confederation_fallback is True


def test_when_extra_field_is_supplied_then_feature_build_config_ignores_it():
    # extra="ignore" — unknown fields must be silently discarded, not raise
    cfg = FeatureBuildConfig(**{"__unexpected_extra__": "should_be_dropped"})
    assert not hasattr(cfg, "__unexpected_extra__")


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


# ── Criterion 2 — AppConfig wiring ─────────────────────────────────────────


def test_when_app_config_is_default_then_features_build_is_feature_build_config():
    assert isinstance(AppConfig().features_build, FeatureBuildConfig)


def test_when_app_config_features_build_is_inspected_then_defaults_match():
    cfg = AppConfig().features_build
    assert cfg.ranking_staleness_cutoff == "2020-12-10"
    assert cfg.form_window == 5
    assert cfg.form_half_life_days == 365.0
    assert cfg.random_seed == 42
    assert cfg.confederation_fallback is True


# ── Criterion 3 — CONFEDERATIONS tuple + CONFEDERATION_MAP coverage ────────


def test_when_confederations_is_inspected_then_it_is_a_tuple():
    assert isinstance(CONFEDERATIONS, tuple)


def test_when_confederations_is_inspected_then_it_has_six_members():
    assert len(CONFEDERATIONS) == 6


def test_when_confederations_is_inspected_then_it_contains_all_six_bodies():
    assert set(CONFEDERATIONS) == {"UEFA", "CONMEBOL", "CAF", "AFC", "CONCACAF", "OFC"}


def test_when_confederations_is_inspected_then_order_matches_the_spec():
    assert CONFEDERATIONS == ("UEFA", "CONMEBOL", "CAF", "AFC", "CONCACAF", "OFC")


def test_when_confederation_map_is_inspected_then_it_has_at_least_48_entries():
    assert len(CONFEDERATION_MAP) >= 48


def test_when_confederation_map_values_are_inspected_then_all_belong_to_confederations():
    for team, conf in CONFEDERATION_MAP.items():
        assert conf in CONFEDERATIONS, f"{team!r} mapped to {conf!r} which is not in CONFEDERATIONS"


def test_when_confederation_map_keys_are_normalized_then_round_trip_holds():
    """Every key must already be crosswalk-canonical so post-normalization lookups hit."""
    non_canonical = [k for k in CONFEDERATION_MAP if normalize_team(k) != k]
    assert non_canonical == [], f"Non-canonical keys in CONFEDERATION_MAP: {non_canonical}"


# Parametrised WC2026 spot-check — one representative per confederation
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


# ── Criterion 4 — confederation_of ─────────────────────────────────────────


def test_when_known_team_is_given_then_confederation_of_returns_its_confederation():
    assert confederation_of("France") == "UEFA"


def test_when_turkiye_alias_is_queried_then_uefa_is_returned():
    # "Türkiye" is a crosswalk alias that normalizes to "Turkey" → UEFA
    assert confederation_of("Türkiye") == "UEFA"


def test_when_unknown_team_is_given_then_confederation_of_returns_none():
    assert confederation_of("Atlantis FC") is None


def test_when_empty_string_is_given_then_confederation_of_returns_none_without_raising():
    result = confederation_of("")
    assert result is None


@given(st.text())
@settings(max_examples=200)
def test_when_any_string_is_given_then_confederation_of_never_raises(name: str) -> None:
    """Total-function invariant: confederation_of must not raise for any str input."""
    result = confederation_of(name)
    assert result is None or result in CONFEDERATIONS


@given(st.sampled_from(sorted(CONFEDERATION_MAP)))
def test_when_confederation_map_key_is_normalised_then_it_is_unchanged(team: str) -> None:
    """Round-trip invariant: every key in CONFEDERATION_MAP is already canonical."""
    assert normalize_team(team) == team


# ── Criterion 5 — resolve_ranking ──────────────────────────────────────────

# The 'ranking' parameter is a dict {team: value} or None.
# "team-absent" means the queried team is missing from the dict.


def test_when_resolve_ranking_returns_result_then_it_has_required_attributes():
    res = resolve_ranking(
        "France", ranking={"France": 10.0}, as_of="2020-01-01", staleness_cutoff="2020-12-10"
    )
    assert hasattr(res, "value")
    assert hasattr(res, "confederation")
    assert hasattr(res, "used_fallback")


def test_when_ranking_is_present_and_date_is_fresh_then_used_fallback_is_false():
    res = resolve_ranking(
        "France", ranking={"France": 1850.0}, as_of="2020-01-01", staleness_cutoff="2020-12-10"
    )
    assert res.used_fallback is False


def test_when_ranking_is_present_and_date_is_fresh_then_value_is_the_ranking():
    res = resolve_ranking(
        "France", ranking={"France": 1850.0}, as_of="2020-01-01", staleness_cutoff="2020-12-10"
    )
    assert res.value == 1850.0


def test_when_ranking_is_present_and_date_is_fresh_then_confederation_is_populated():
    res = resolve_ranking(
        "France", ranking={"France": 1850.0}, as_of="2020-01-01", staleness_cutoff="2020-12-10"
    )
    assert res.confederation == "UEFA"


def test_when_ranking_is_none_then_used_fallback_is_true():
    res = resolve_ranking("Brazil", ranking=None, as_of="2020-12-10", staleness_cutoff="2020-12-10")
    assert isinstance(res, RankingResolution)
    assert res.used_fallback is True


def test_when_ranking_is_none_then_value_is_none():
    res = resolve_ranking("Brazil", ranking=None, as_of="2020-12-10", staleness_cutoff="2020-12-10")
    assert res.value is None


def test_when_ranking_is_none_then_confederation_is_still_populated():
    res = resolve_ranking("Brazil", ranking=None, as_of="2020-12-10", staleness_cutoff="2020-12-10")
    assert res.confederation == "CONMEBOL"


def test_when_as_of_is_after_cutoff_then_used_fallback_is_true():
    # as_of > staleness_cutoff → stale → fallback
    res = resolve_ranking(
        "Brazil", ranking={"Brazil": 1750.5}, as_of="2021-01-01", staleness_cutoff="2020-12-10"
    )
    assert res.used_fallback is True


def test_when_as_of_is_after_cutoff_then_value_is_none():
    res = resolve_ranking(
        "Brazil", ranking={"Brazil": 1750.5}, as_of="2021-01-01", staleness_cutoff="2020-12-10"
    )
    assert res.value is None


def test_when_as_of_is_after_cutoff_then_confederation_is_still_populated():
    res = resolve_ranking(
        "Brazil", ranking={"Brazil": 1750.5}, as_of="2021-01-01", staleness_cutoff="2020-12-10"
    )
    assert res.confederation == "CONMEBOL"


def test_when_as_of_equals_cutoff_then_ranking_is_not_stale():
    """Boundary: as_of == staleness_cutoff is fresh, not stale (only as_of > cutoff is stale)."""
    res = resolve_ranking(
        "France", ranking={"France": 1850.0}, as_of="2020-12-10", staleness_cutoff="2020-12-10"
    )
    assert res.used_fallback is False
    assert res.value == 1850.0


def test_when_as_of_is_before_cutoff_then_fallback_is_not_taken():
    res = resolve_ranking(
        "France", ranking={"France": 1850.0}, as_of="2020-06-01", staleness_cutoff="2020-12-10"
    )
    assert res.used_fallback is False
    assert res.value == 1850.0


def test_when_team_is_absent_from_ranking_dict_then_used_fallback_is_true():
    res = resolve_ranking(
        "Japan", ranking={"France": 1800.0}, as_of="2020-06-01", staleness_cutoff="2020-12-10"
    )
    assert res.used_fallback is True


def test_when_team_is_absent_from_ranking_dict_then_value_is_none():
    res = resolve_ranking(
        "Japan", ranking={"France": 1800.0}, as_of="2020-06-01", staleness_cutoff="2020-12-10"
    )
    assert res.value is None


def test_when_team_is_absent_from_ranking_dict_then_confederation_is_populated():
    res = resolve_ranking(
        "Japan", ranking={"France": 1800.0}, as_of="2020-06-01", staleness_cutoff="2020-12-10"
    )
    assert res.confederation == "AFC"


def test_when_ranking_is_none_and_team_is_afc_then_confederation_is_afc():
    res = resolve_ranking("Japan", ranking=None, as_of="2020-12-10", staleness_cutoff="2020-12-10")
    assert res.confederation == "AFC"


# ── Property-based: staleness invariant ────────────────────────────────────


@given(
    st.dates(
        min_value=__import__("datetime").date(2021, 1, 1),
        max_value=__import__("datetime").date(2030, 12, 31),
    ).map(str)
)
@settings(max_examples=100)
def test_when_as_of_is_strictly_after_cutoff_then_used_fallback_is_always_true(as_of: str) -> None:
    """Invariant: any ISO date after 2020-12-10 always triggers fallback for a known team."""
    res = resolve_ranking(
        "France", ranking={"France": 1850.0}, as_of=as_of, staleness_cutoff="2020-12-10"
    )
    assert res.used_fallback is True
    assert res.value is None
    assert res.confederation is not None
