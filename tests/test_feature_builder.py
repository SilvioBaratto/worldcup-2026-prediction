"""Source-blind example tests for worldcup_playoff.features.build (Issue #9).

All tests are derived solely from the acceptance criteria and requirements.md.
No implementation source was read during authorship (Red-phase TDD).

Criteria covered (UNIT-verifiable only; NOT VERIFIABLE criteria skipped per oracle):
  1  Output frame has exactly one row per input match; unplayed goals pass through as <NA>.
  2  elo_diff == home_elo − away_elo (consumed, not recomputed); attack/defence columns
     equal the hand-built TeamAbilities passed in (no refit).
  3  Neutral flag round-trips exactly from the input NEUTRAL column.
  4  Misaligned elo_df / matches_df raises ValueError.
  5  Confederation columns deterministically encoded over the six CONFEDERATIONS; a None
     ranking degrades to confederation without breaking the output frame.
  6  Output is deterministic given FeatureBuildConfig.random_seed.

Criteria skipped (NOT VERIFIABLE per oracle):
  - Football-only allow-list: exact output column set
  - No-leakage end-to-end test (two-match history)
  - "All tests pass" / SOLID quality gates
"""

from __future__ import annotations

import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from worldcup_playoff.config import FeatureBuildConfig
from worldcup_playoff.features.build import FeatureBuilder, TeamAbilities
from worldcup_playoff.features.confederation import CONFEDERATIONS

# ---------------------------------------------------------------------------
# Shared helpers — built from acceptance criteria and data contracts only
# ---------------------------------------------------------------------------

_DEFAULT_CFG = FeatureBuildConfig(random_seed=42)


def _matches(rows: list[dict]) -> pd.DataFrame:
    """Minimal matches DataFrame in the martj42-derived internal schema."""
    return pd.DataFrame(
        {
            "DATE": pd.to_datetime([r.get("date", "2026-01-01") for r in rows]),
            "HOME_TEAM": [r["home"] for r in rows],
            "AWAY_TEAM": [r["away"] for r in rows],
            "HOME_GOALS": pd.array([r.get("home_goals", pd.NA) for r in rows], dtype="Int64"),
            "AWAY_GOALS": pd.array([r.get("away_goals", pd.NA) for r in rows], dtype="Int64"),
            "TOURNAMENT": [r.get("tournament", "Friendly") for r in rows],
            "NEUTRAL": [r.get("neutral", False) for r in rows],
        }
    )


def _elo(rows: list[dict]) -> pd.DataFrame:
    """Pre-computed Elo DataFrame aligned row-for-row with the matches."""
    return pd.DataFrame(
        {
            "home_elo": [float(r.get("home_elo", 1500)) for r in rows],
            "away_elo": [float(r.get("away_elo", 1500)) for r in rows],
        }
    )


def _abilities(teams: dict[str, dict[str, float]]) -> TeamAbilities:
    """Build a hand-crafted TeamAbilities from {team: {"attack": f, "defence": f}}."""
    return TeamAbilities(
        attack={name: v["attack"] for name, v in teams.items()},
        defence={name: v["defence"] for name, v in teams.items()},
        home_adv=0.0,
        rho=-0.1,
        intercept=0.0,
    )


# Pre-built ability sets reused across multiple tests
_BRA_GER = _abilities(
    {
        "Brazil": {"attack": 1.50, "defence": 0.80},
        "Germany": {"attack": 1.30, "defence": 0.90},
    }
)

_BRA_GER_FRA_ESP = _abilities(
    {
        "Brazil": {"attack": 1.50, "defence": 0.80},
        "Germany": {"attack": 1.30, "defence": 0.90},
        "France": {"attack": 1.20, "defence": 0.85},
        "Spain": {"attack": 1.40, "defence": 0.75},
    }
)


# ---------------------------------------------------------------------------
# Criterion 1 — one row per input match; unplayed labels pass as <NA>
# ---------------------------------------------------------------------------


class TestOutputShapeAndLabels:
    """Output frame must have exactly one row per input match (including unplayed WC2026 rows).

    Labels home_goals / away_goals pass through as <NA> when the source goals are absent
    (unplayed fixtures whose scores are NA in the martj42 dataset).
    """

    def test_when_two_played_matches_provided_then_output_has_two_rows(self) -> None:
        m = _matches(
            [
                {"home": "Brazil", "away": "Germany", "home_goals": 2, "away_goals": 1},
                {"home": "France", "away": "Spain", "home_goals": 0, "away_goals": 0},
            ]
        )
        result = FeatureBuilder(_DEFAULT_CFG).build(m, _elo([{}, {}]), _BRA_GER_FRA_ESP)
        assert len(result) == 2

    def test_when_three_matches_include_unplayed_then_all_three_rows_appear_in_output(self) -> None:
        """Unplayed WC2026 fixtures must NOT be dropped."""
        m = _matches(
            [
                {"home": "Brazil", "away": "Germany", "home_goals": 2, "away_goals": 1},
                {"home": "France", "away": "Spain"},  # unplayed
                {"home": "Germany", "away": "France"},  # unplayed
            ]
        )
        result = FeatureBuilder(_DEFAULT_CFG).build(m, _elo([{}, {}, {}]), _BRA_GER_FRA_ESP)
        assert len(result) == 3

    def test_when_match_is_unplayed_then_home_goals_column_is_na(self) -> None:
        m = _matches([{"home": "Brazil", "away": "Germany"}])  # no goals → NA
        result = FeatureBuilder(_DEFAULT_CFG).build(m, _elo([{}]), _BRA_GER)
        assert pd.isna(result.iloc[0]["home_goals"])

    def test_when_match_is_unplayed_then_away_goals_column_is_na(self) -> None:
        m = _matches([{"home": "Brazil", "away": "Germany"}])
        result = FeatureBuilder(_DEFAULT_CFG).build(m, _elo([{}]), _BRA_GER)
        assert pd.isna(result.iloc[0]["away_goals"])

    def test_when_match_is_played_then_home_goals_label_passes_through(self) -> None:
        m = _matches([{"home": "Brazil", "away": "Germany", "home_goals": 3, "away_goals": 0}])
        result = FeatureBuilder(_DEFAULT_CFG).build(m, _elo([{}]), _BRA_GER)
        assert result.iloc[0]["home_goals"] == 3

    def test_when_match_is_played_then_away_goals_label_passes_through(self) -> None:
        m = _matches([{"home": "Brazil", "away": "Germany", "home_goals": 3, "away_goals": 0}])
        result = FeatureBuilder(_DEFAULT_CFG).build(m, _elo([{}]), _BRA_GER)
        assert result.iloc[0]["away_goals"] == 0


# ---------------------------------------------------------------------------
# Criterion 2 — elo_diff == home_elo − away_elo; abilities consumed, not refit
# ---------------------------------------------------------------------------


class TestEloDiffAndAbilities:
    """elo_diff must equal home_elo − away_elo for every row (consumed from elo_df).

    Attack and defence output columns must equal the values in the hand-built TeamAbilities
    passed in — proving the builder did NOT refit abilities from match data.
    """

    def test_when_home_elo_1600_and_away_1400_then_elo_diff_is_200(self) -> None:
        m = _matches([{"home": "Brazil", "away": "Germany", "home_goals": 1, "away_goals": 0}])
        result = FeatureBuilder(_DEFAULT_CFG).build(
            m, _elo([{"home_elo": 1600.0, "away_elo": 1400.0}]), _BRA_GER
        )
        assert result.iloc[0]["elo_diff"] == pytest.approx(200.0)

    def test_when_away_elo_exceeds_home_then_elo_diff_is_negative(self) -> None:
        m = _matches([{"home": "Germany", "away": "Brazil", "home_goals": 0, "away_goals": 1}])
        result = FeatureBuilder(_DEFAULT_CFG).build(
            m,
            _elo([{"home_elo": 1400.0, "away_elo": 1600.0}]),
            _abilities(
                {
                    "Germany": {"attack": 1.3, "defence": 0.9},
                    "Brazil": {"attack": 1.5, "defence": 0.8},
                }
            ),
        )
        assert result.iloc[0]["elo_diff"] == pytest.approx(-200.0)

    def test_when_multiple_rows_then_every_elo_diff_equals_home_minus_away(self) -> None:
        m = _matches(
            [
                {"home": "Brazil", "away": "Germany", "home_goals": 2, "away_goals": 1},
                {"home": "France", "away": "Spain", "home_goals": 0, "away_goals": 0},
            ]
        )
        result = FeatureBuilder(_DEFAULT_CFG).build(
            m,
            _elo(
                [{"home_elo": 1600.0, "away_elo": 1400.0}, {"home_elo": 1550.0, "away_elo": 1580.0}]
            ),
            _BRA_GER_FRA_ESP,
        )
        assert result.iloc[0]["elo_diff"] == pytest.approx(200.0)
        assert result.iloc[1]["elo_diff"] == pytest.approx(-30.0)

    def test_when_abilities_passed_in_then_home_attack_column_matches_input(self) -> None:
        abilities = TeamAbilities(
            attack={"Brazil": 1.789, "Germany": 1.345},
            defence={"Brazil": 0.832, "Germany": 0.912},
            home_adv=0.0,
            rho=-0.1,
            intercept=0.0,
        )
        m = _matches([{"home": "Brazil", "away": "Germany", "home_goals": 1, "away_goals": 0}])
        result = FeatureBuilder(_DEFAULT_CFG).build(m, _elo([{}]), abilities)
        assert result.iloc[0]["home_attack"] == pytest.approx(1.789)

    def test_when_abilities_passed_in_then_away_attack_column_matches_input(self) -> None:
        abilities = TeamAbilities(
            attack={"Brazil": 1.789, "Germany": 1.345},
            defence={"Brazil": 0.832, "Germany": 0.912},
            home_adv=0.0,
            rho=-0.1,
            intercept=0.0,
        )
        m = _matches([{"home": "Brazil", "away": "Germany", "home_goals": 1, "away_goals": 0}])
        result = FeatureBuilder(_DEFAULT_CFG).build(m, _elo([{}]), abilities)
        assert result.iloc[0]["away_attack"] == pytest.approx(1.345)

    def test_when_abilities_passed_in_then_home_defence_column_matches_input(self) -> None:
        abilities = TeamAbilities(
            attack={"Brazil": 1.789, "Germany": 1.345},
            defence={"Brazil": 0.832, "Germany": 0.912},
            home_adv=0.0,
            rho=-0.1,
            intercept=0.0,
        )
        m = _matches([{"home": "Brazil", "away": "Germany", "home_goals": 1, "away_goals": 0}])
        result = FeatureBuilder(_DEFAULT_CFG).build(m, _elo([{}]), abilities)
        assert result.iloc[0]["home_defence"] == pytest.approx(0.832)

    def test_when_abilities_passed_in_then_away_defence_column_matches_input(self) -> None:
        abilities = TeamAbilities(
            attack={"Brazil": 1.789, "Germany": 1.345},
            defence={"Brazil": 0.832, "Germany": 0.912},
            home_adv=0.0,
            rho=-0.1,
            intercept=0.0,
        )
        m = _matches([{"home": "Brazil", "away": "Germany", "home_goals": 1, "away_goals": 0}])
        result = FeatureBuilder(_DEFAULT_CFG).build(m, _elo([{}]), abilities)
        assert result.iloc[0]["away_defence"] == pytest.approx(0.912)


# ---------------------------------------------------------------------------
# Criterion 3 — neutral flag round-trips from the NEUTRAL column
# ---------------------------------------------------------------------------


class TestNeutralFlagRoundTrip:
    """The output neutral column must preserve the NEUTRAL input value exactly."""

    def test_when_neutral_is_true_then_output_neutral_is_truthy(self) -> None:
        m = _matches(
            [
                {
                    "home": "Brazil",
                    "away": "Germany",
                    "home_goals": 1,
                    "away_goals": 0,
                    "neutral": True,
                }
            ]
        )
        result = FeatureBuilder(_DEFAULT_CFG).build(m, _elo([{}]), _BRA_GER)
        assert bool(result.iloc[0]["neutral"]) is True

    def test_when_neutral_is_false_then_output_neutral_is_falsy(self) -> None:
        m = _matches(
            [
                {
                    "home": "Brazil",
                    "away": "Germany",
                    "home_goals": 1,
                    "away_goals": 0,
                    "neutral": False,
                }
            ]
        )
        result = FeatureBuilder(_DEFAULT_CFG).build(m, _elo([{}]), _BRA_GER)
        assert bool(result.iloc[0]["neutral"]) is False


# ---------------------------------------------------------------------------
# Criterion 4 — misaligned elo_df raises ValueError
# ---------------------------------------------------------------------------


class TestAlignmentGuard:
    """elo_df and matches_df must have the same length; any mismatch raises ValueError."""

    def test_when_elo_df_shorter_than_matches_then_value_error_is_raised(self) -> None:
        m = _matches(
            [
                {"home": "Brazil", "away": "Germany", "home_goals": 2, "away_goals": 1},
                {"home": "France", "away": "Spain", "home_goals": 0, "away_goals": 0},
            ]
        )
        short_elo = _elo([{}])  # 1 row vs 2 matches
        with pytest.raises(ValueError):
            FeatureBuilder(_DEFAULT_CFG).build(m, short_elo, _BRA_GER_FRA_ESP)

    def test_when_elo_df_longer_than_matches_then_value_error_is_raised(self) -> None:
        m = _matches([{"home": "Brazil", "away": "Germany", "home_goals": 2, "away_goals": 1}])
        long_elo = _elo([{}, {}])  # 2 rows vs 1 match
        with pytest.raises(ValueError):
            FeatureBuilder(_DEFAULT_CFG).build(m, long_elo, _BRA_GER)


# ---------------------------------------------------------------------------
# Criterion 5 — confederation deterministic over 6 codes; None ranking OK
# ---------------------------------------------------------------------------


class TestConfederationEncoding:
    """Confederation columns are deterministically encoded over the six CONFEDERATIONS.

    When the FIFA ranking DataFrame is None, the builder must fall back to the static
    confederation map and still populate confederation columns — no crash, no NA.
    """

    def test_when_home_team_is_brazil_then_home_confederation_is_conmebol(self) -> None:
        m = _matches([{"home": "Brazil", "away": "Germany", "home_goals": 2, "away_goals": 1}])
        result = FeatureBuilder(_DEFAULT_CFG).build(m, _elo([{}]), _BRA_GER)
        assert result.iloc[0]["home_confederation"] == "CONMEBOL"

    def test_when_home_team_is_germany_then_home_confederation_is_uefa(self) -> None:
        m = _matches([{"home": "Germany", "away": "France", "home_goals": 1, "away_goals": 1}])
        result = FeatureBuilder(_DEFAULT_CFG).build(
            m,
            _elo([{}]),
            _abilities(
                {
                    "Germany": {"attack": 1.3, "defence": 0.9},
                    "France": {"attack": 1.2, "defence": 0.85},
                }
            ),
        )
        assert result.iloc[0]["home_confederation"] == "UEFA"

    def test_when_away_team_is_brazil_then_away_confederation_is_conmebol(self) -> None:
        m = _matches([{"home": "Germany", "away": "Brazil", "home_goals": 0, "away_goals": 1}])
        result = FeatureBuilder(_DEFAULT_CFG).build(
            m,
            _elo([{}]),
            _abilities(
                {
                    "Germany": {"attack": 1.3, "defence": 0.9},
                    "Brazil": {"attack": 1.5, "defence": 0.8},
                }
            ),
        )
        assert result.iloc[0]["away_confederation"] == "CONMEBOL"

    def test_when_ranking_df_is_none_then_home_confederation_column_is_populated(self) -> None:
        """None FIFA ranking degrades gracefully — static map provides confederation."""
        m = _matches([{"home": "Brazil", "away": "Germany", "home_goals": 2, "away_goals": 1}])
        result = FeatureBuilder(_DEFAULT_CFG).build(m, _elo([{}]), _BRA_GER, ranking_df=None)
        assert "home_confederation" in result.columns
        assert not pd.isna(result.iloc[0]["home_confederation"])

    def test_when_ranking_df_is_none_then_away_confederation_column_is_populated(self) -> None:
        m = _matches([{"home": "Brazil", "away": "Germany", "home_goals": 2, "away_goals": 1}])
        result = FeatureBuilder(_DEFAULT_CFG).build(m, _elo([{}]), _BRA_GER, ranking_df=None)
        assert "away_confederation" in result.columns
        assert not pd.isna(result.iloc[0]["away_confederation"])

    def test_when_same_input_run_twice_then_confederation_output_is_identical(self) -> None:
        """Determinism: same input always yields the same confederation encoding."""
        m = _matches([{"home": "Brazil", "away": "Argentina", "home_goals": 1, "away_goals": 0}])
        ab = _abilities(
            {
                "Brazil": {"attack": 1.5, "defence": 0.8},
                "Argentina": {"attack": 1.6, "defence": 0.7},
            }
        )
        r1 = FeatureBuilder(_DEFAULT_CFG).build(m, _elo([{}]), ab)
        r2 = FeatureBuilder(_DEFAULT_CFG).build(m, _elo([{}]), ab)
        assert r1.iloc[0]["home_confederation"] == r2.iloc[0]["home_confederation"]
        assert r1.iloc[0]["away_confederation"] == r2.iloc[0]["away_confederation"]


# ---------------------------------------------------------------------------
# Criterion 6 — deterministic given FeatureBuildConfig.random_seed
# ---------------------------------------------------------------------------


class TestDeterminism:
    """FeatureBuilder output is byte-identical when FeatureBuildConfig.random_seed is fixed."""

    def test_when_same_seed_then_build_produces_identical_frames(self) -> None:
        m = _matches(
            [
                {"home": "Brazil", "away": "Germany", "home_goals": 2, "away_goals": 1},
                {"home": "France", "away": "Spain", "home_goals": 0, "away_goals": 0},
            ]
        )
        elo = _elo(
            [{"home_elo": 1600.0, "away_elo": 1400.0}, {"home_elo": 1550.0, "away_elo": 1580.0}]
        )

        cfg_a = FeatureBuildConfig(random_seed=42)
        cfg_b = FeatureBuildConfig(random_seed=42)

        result_a = FeatureBuilder(cfg_a).build(m, elo, _BRA_GER_FRA_ESP)
        result_b = FeatureBuilder(cfg_b).build(m, elo, _BRA_GER_FRA_ESP)

        pd.testing.assert_frame_equal(result_a, result_b)


# ---------------------------------------------------------------------------
# Property-based tests (hypothesis)
# ---------------------------------------------------------------------------


@given(st.booleans())
@settings(max_examples=10)
def test_when_neutral_flag_is_any_boolean_then_it_round_trips_through_features(
    neutral: bool,
) -> None:
    """Round-trip invariant: the NEUTRAL bool is preserved exactly in the output column.

    Derived from criterion: 'Neutral flag round-trips exactly from the input NEUTRAL column.'
    """
    m = _matches(
        [
            {
                "home": "Brazil",
                "away": "Germany",
                "home_goals": 1,
                "away_goals": 0,
                "neutral": neutral,
            }
        ]
    )
    result = FeatureBuilder(_DEFAULT_CFG).build(m, _elo([{}]), _BRA_GER)
    assert bool(result.iloc[0]["neutral"]) == neutral


@given(st.integers(min_value=1, max_value=8))
@settings(max_examples=20)
def test_when_n_matches_are_provided_then_output_has_exactly_n_rows(n: int) -> None:
    """Count invariant: len(output) == len(input) for every valid n.

    Derived from criterion: 'Output frame has exactly one row per input match.'
    """
    rows = [
        {
            "home": "Brazil",
            "away": "Germany",
            "date": f"2026-01-{i + 1:02d}",
            "home_goals": 1,
            "away_goals": 0,
        }
        for i in range(n)
    ]
    result = FeatureBuilder(_DEFAULT_CFG).build(_matches(rows), _elo([{}] * n), _BRA_GER)
    assert len(result) == n


@given(
    st.sampled_from(
        [
            "Brazil",
            "Argentina",
            "Germany",
            "France",
            "Spain",
            "Japan",
            "Mexico",
            "United States",
            "Senegal",
            "New Zealand",
        ]
    )
)
@settings(max_examples=10)
def test_when_confederation_is_built_for_any_known_team_then_value_is_one_of_six_valid_codes(
    home_team: str,
) -> None:
    """Valid-output-domain invariant: home_confederation ∈ CONFEDERATIONS for all known teams.

    Derived from criterion: 'Confederation columns are deterministically encoded over the
    six CONFEDERATIONS indicators.'
    """
    away_team = "Germany" if home_team != "Germany" else "Brazil"
    m = _matches([{"home": home_team, "away": away_team, "home_goals": 1, "away_goals": 0}])
    result = FeatureBuilder(_DEFAULT_CFG).build(
        m,
        _elo([{}]),
        _abilities(
            {
                home_team: {"attack": 1.2, "defence": 0.9},
                away_team: {"attack": 1.1, "defence": 0.95},
            }
        ),
    )
    assert result.iloc[0]["home_confederation"] in CONFEDERATIONS
    assert result.iloc[0]["away_confederation"] in CONFEDERATIONS
