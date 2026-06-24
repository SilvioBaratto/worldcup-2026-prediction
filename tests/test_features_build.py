"""Source-blind example tests for worldcup_playoff/features/build.py.

Covers Issue #9 criteria (originally) and all verifiable criteria from Issue
#36 acceptance criteria.  All behaviour is derived from the acceptance-criteria
text only — no implementation source was read during authoring (Red-phase TDD).

Oracle classifications for Issue #36:
  [UNIT] AC1 — FEATURE_COLUMNS tuple contract + to_frame() dtype/order
  [UNIT] AC2 — FeatureBuilder.build() row-count, NA-goals, elo_diff
  [UNIT] AC3 — build_features() EloResult duck-type, tournament column
  [UNIT] AC4 — Alignment guard ValueError
  [UNIT] AC5 — End-to-end no-leakage + determinism
  [NOT VERIFIABLE] AC6 — All tests pass (suite gate)
  [NOT VERIFIABLE] AC7 — SOLID / code quality

Invariant choices documented where criterion text is ambiguous:
- "home_form" is a time-weighted PPG score on the 3/1/0 scale; the first
  chronological match for a team has no prior record → 0.0.
- "unplayed fixture" = row where HOME_GOALS / AWAY_GOALS are NaN / pd.NA.
- EloResult is any object with a .match_diffs attribute that returns a
  DataFrame pre-aligned with sort_chronological(df).
- TeamAbilities has at minimum attack/defence per-team dicts; home_adv, rho,
  intercept are passed as floats consistent with Dixon-Coles spec.
"""

from __future__ import annotations

import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from worldcup_playoff.config import FeatureBuildConfig
from worldcup_playoff.features.build import (
    FEATURE_COLUMNS,
    FeatureBuilder,
    TeamAbilities,
    build_features,
)

_CFG = FeatureBuildConfig(random_seed=42)

_FORBIDDEN_SUBSTRINGS = ("gdp", "market_value", "transfer", "odds", "bookie", "wage")


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------


def _matches(rows: list[dict]) -> pd.DataFrame:
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
    return pd.DataFrame(
        {
            "home_elo": [float(r.get("home_elo", 1500)) for r in rows],
            "away_elo": [float(r.get("away_elo", 1500)) for r in rows],
        }
    )


def _team_abilities(*teams: str) -> TeamAbilities:
    """Uniform stub TeamAbilities for the given set of team names."""
    return TeamAbilities(
        attack={t: 1.0 for t in teams},
        defence={t: 1.0 for t in teams},
        home_adv=0.0,
        rho=-0.1,
        intercept=0.0,
    )


_BRA_GER = TeamAbilities(
    attack={"Brazil": 1.5, "Germany": 1.3},
    defence={"Brazil": 0.8, "Germany": 0.9},
    home_adv=0.0,
    rho=-0.1,
    intercept=0.0,
)

_BRA_FRA = TeamAbilities(
    attack={"Brazil": 1.5, "France": 1.4},
    defence={"Brazil": 0.8, "France": 0.9},
    home_adv=0.0,
    rho=-0.1,
    intercept=0.0,
)

_MULTI = _team_abilities("Brazil", "France", "Germany", "Spain", "Argentina")


class _FakeEloResult:
    """Duck-typed stand-in for EloResult — exposes only .match_diffs."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.match_diffs = df


# ---------------------------------------------------------------------------
# AC1 — FEATURE_COLUMNS tuple contract + to_frame() dtype / column order
# ---------------------------------------------------------------------------


class TestFeatureColumnsTupleContract:
    """FEATURE_COLUMNS is the exact stable tuple; to_frame() reindexes to it."""

    def test_when_feature_columns_inspected_then_it_is_a_tuple(self) -> None:
        assert isinstance(FEATURE_COLUMNS, tuple)

    def test_when_feature_columns_inspected_then_home_goals_is_present(self) -> None:
        assert "home_goals" in FEATURE_COLUMNS

    def test_when_feature_columns_inspected_then_away_goals_is_present(self) -> None:
        assert "away_goals" in FEATURE_COLUMNS

    def test_when_feature_columns_inspected_then_elo_diff_is_present(self) -> None:
        assert "elo_diff" in FEATURE_COLUMNS

    @pytest.mark.parametrize("forbidden", _FORBIDDEN_SUBSTRINGS)
    def test_when_feature_columns_inspected_then_no_forbidden_substring_is_present(
        self, forbidden: str
    ) -> None:
        offending = [c for c in FEATURE_COLUMNS if forbidden in c.lower()]
        assert not offending, f"Columns {offending!r} contain forbidden substring {forbidden!r}"

    def test_when_features_are_built_then_column_set_equals_feature_columns(self) -> None:
        m = _matches([{"home": "Brazil", "away": "Germany", "home_goals": 1, "away_goals": 0}])
        result = FeatureBuilder(_CFG).build(m, _elo([{}]), _BRA_GER)
        assert set(result.columns) == set(FEATURE_COLUMNS)

    def test_when_features_are_built_then_column_order_matches_feature_columns(self) -> None:
        m = _matches([{"home": "Brazil", "away": "Germany", "home_goals": 1, "away_goals": 0}])
        result = FeatureBuilder(_CFG).build(m, _elo([{}]), _BRA_GER)
        assert list(result.columns) == list(FEATURE_COLUMNS)

    def test_when_features_are_built_then_no_forbidden_column_names_are_present(self) -> None:
        m = _matches([{"home": "Brazil", "away": "Germany", "home_goals": 1, "away_goals": 0}])
        result = FeatureBuilder(_CFG).build(m, _elo([{}]), _BRA_GER)
        for col in result.columns:
            for substr in _FORBIDDEN_SUBSTRINGS:
                assert substr not in col.lower(), f"Forbidden column found: {col!r}"

    def test_when_to_frame_produces_output_then_home_goals_dtype_is_int64(self) -> None:
        """to_frame() must cast home_goals to pandas nullable Int64."""
        m = _matches([{"home": "Brazil", "away": "Germany", "home_goals": 2, "away_goals": 1}])
        result = FeatureBuilder(_CFG).build(m, _elo([{}]), _BRA_GER)
        assert result["home_goals"].dtype == pd.Int64Dtype()

    def test_when_to_frame_produces_output_then_away_goals_dtype_is_int64(self) -> None:
        """to_frame() must cast away_goals to pandas nullable Int64."""
        m = _matches([{"home": "Brazil", "away": "Germany", "home_goals": 2, "away_goals": 1}])
        result = FeatureBuilder(_CFG).build(m, _elo([{}]), _BRA_GER)
        assert result["away_goals"].dtype == pd.Int64Dtype()

    def test_when_unplayed_fixture_goals_flow_through_to_frame_then_dtype_is_still_int64(
        self,
    ) -> None:
        """Nullable Int64 preserves <NA> for unplayed fixtures — dtype must not widen to float."""
        m = _matches(
            [{"home": "Brazil", "away": "Germany", "neutral": True, "tournament": "FIFA World Cup"}]
        )
        result = FeatureBuilder(_CFG).build(m, _elo([{}]), _BRA_GER)
        assert result["home_goals"].dtype == pd.Int64Dtype()
        assert result["away_goals"].dtype == pd.Int64Dtype()


# ---------------------------------------------------------------------------
# AC2 — FeatureBuilder.build() contracts
# ---------------------------------------------------------------------------


class TestFeatureBuilderBuild:
    """FeatureBuilder.build(df, elo_df, abilities) emits one row per input match."""

    def test_when_build_called_with_one_match_then_one_row_is_emitted(self) -> None:
        m = _matches([{"home": "Brazil", "away": "France", "home_goals": 2, "away_goals": 1}])
        result = FeatureBuilder(_CFG).build(m, _elo([{}]), _BRA_FRA)
        assert len(result) == 1

    def test_when_build_called_with_two_matches_then_two_rows_are_emitted(self) -> None:
        m = _matches(
            [
                {
                    "home": "Brazil",
                    "away": "France",
                    "date": "2024-01-01",
                    "home_goals": 2,
                    "away_goals": 1,
                },
                {
                    "home": "Germany",
                    "away": "Spain",
                    "date": "2024-01-15",
                    "home_goals": 1,
                    "away_goals": 0,
                },
            ]
        )
        ab = _team_abilities("Brazil", "France", "Germany", "Spain")
        result = FeatureBuilder(_CFG).build(m, _elo([{}, {}]), ab)
        assert len(result) == 2

    def test_when_build_called_with_five_matches_then_five_rows_are_emitted(self) -> None:
        rows = [
            {
                "home": "Brazil",
                "away": "France",
                "date": f"2024-01-{i:02d}",
                "home_goals": i % 4,
                "away_goals": 0,
            }
            for i in range(1, 6)
        ]
        m = _matches(rows)
        result = FeatureBuilder(_CFG).build(m, _elo([{}] * 5), _BRA_FRA)
        assert len(result) == 5

    def test_when_build_called_with_unplayed_fixture_then_home_goals_is_na(self) -> None:
        """WC2026 unplayed fixtures have <NA> goals — must propagate unchanged."""
        m = _matches(
            [{"home": "Brazil", "away": "Germany", "neutral": True, "tournament": "FIFA World Cup"}]
        )
        result = FeatureBuilder(_CFG).build(m, _elo([{}]), _BRA_GER)
        assert pd.isna(result["home_goals"].iloc[0])

    def test_when_build_called_with_unplayed_fixture_then_away_goals_is_na(self) -> None:
        m = _matches(
            [{"home": "Brazil", "away": "Germany", "neutral": True, "tournament": "FIFA World Cup"}]
        )
        result = FeatureBuilder(_CFG).build(m, _elo([{}]), _BRA_GER)
        assert pd.isna(result["away_goals"].iloc[0])

    def test_when_build_called_then_elo_diff_equals_home_elo_minus_away_elo(self) -> None:
        home_elo, away_elo = 1900.0, 1750.0
        m = _matches([{"home": "Brazil", "away": "Germany", "home_goals": 2, "away_goals": 0}])
        elo = _elo([{"home_elo": home_elo, "away_elo": away_elo}])
        result = FeatureBuilder(_CFG).build(m, elo, _BRA_GER)
        assert result["elo_diff"].iloc[0] == pytest.approx(home_elo - away_elo)


# ---------------------------------------------------------------------------
# AC3 — build_features() public API contracts
# ---------------------------------------------------------------------------


class TestBuildFeaturesPublicApi:
    """build_features(df, elo, abilities, *, config=None, ranking=None)."""

    def test_when_pre_built_elo_dataframe_passed_then_no_error_is_raised(self) -> None:
        m = _matches([{"home": "Brazil", "away": "Germany", "home_goals": 2, "away_goals": 1}])
        build_features(m, _elo([{}]), _BRA_GER)

    def test_when_elo_result_object_passed_then_no_error_is_raised(self) -> None:
        """build_features converts EloResult via .match_diffs — duck-typed."""
        m = _matches([{"home": "Brazil", "away": "Germany", "home_goals": 2, "away_goals": 1}])
        fake = _FakeEloResult(_elo([{}]))
        build_features(m, fake, _BRA_GER)

    def test_when_config_is_none_then_no_error_is_raised(self) -> None:
        m = _matches([{"home": "Brazil", "away": "Germany", "home_goals": 2, "away_goals": 1}])
        build_features(m, _elo([{}]), _BRA_GER, config=None)

    def test_when_ranking_is_none_then_no_error_is_raised(self) -> None:
        m = _matches([{"home": "Brazil", "away": "Germany", "home_goals": 2, "away_goals": 1}])
        build_features(m, _elo([{}]), _BRA_GER, ranking=None)

    def test_when_build_features_called_then_tournament_metadata_column_is_present(
        self,
    ) -> None:
        m = _matches(
            [
                {
                    "home": "Brazil",
                    "away": "Germany",
                    "home_goals": 2,
                    "away_goals": 1,
                    "tournament": "FIFA World Cup",
                }
            ]
        )
        result = build_features(m, _elo([{}]), _BRA_GER)
        assert "tournament" in result.columns

    def test_when_build_features_called_then_tournament_value_matches_input_tournament(
        self,
    ) -> None:
        m = _matches(
            [
                {
                    "home": "Brazil",
                    "away": "Germany",
                    "home_goals": 2,
                    "away_goals": 1,
                    "tournament": "FIFA World Cup",
                }
            ]
        )
        result = build_features(m, _elo([{}]), _BRA_GER)
        assert result["tournament"].iloc[0] == "FIFA World Cup"

    def test_when_unsorted_input_given_then_tournament_column_follows_chronological_order(
        self,
    ) -> None:
        """tournament comes from sort_chronological(df).TOURNAMENT — output is date-ordered."""
        m = _matches(
            [
                {
                    "home": "Germany",
                    "away": "Spain",
                    "date": "2024-01-15",
                    "home_goals": 1,
                    "away_goals": 0,
                    "tournament": "Friendly",
                },
                {
                    "home": "Brazil",
                    "away": "France",
                    "date": "2024-01-01",
                    "home_goals": 2,
                    "away_goals": 1,
                    "tournament": "FIFA World Cup",
                },
            ]
        )
        ab = _team_abilities("Brazil", "France", "Germany", "Spain")
        result = build_features(m, _elo([{}, {}]), ab)
        # After sort_chronological: row 0 = 2024-01-01 (FIFA World Cup)
        assert result["tournament"].iloc[0] == "FIFA World Cup"
        assert result["tournament"].iloc[1] == "Friendly"


# ---------------------------------------------------------------------------
# AC4 — Alignment guard
# ---------------------------------------------------------------------------


class TestAlignmentGuard:
    """Alignment guard raises ValueError when len(elo_df) != len(sort_chronological(df))."""

    def test_when_elo_has_fewer_rows_than_sorted_df_then_value_error_is_raised(
        self,
    ) -> None:
        m = _matches(
            [
                {
                    "home": "Brazil",
                    "away": "France",
                    "date": "2024-01-01",
                    "home_goals": 2,
                    "away_goals": 1,
                },
                {
                    "home": "Germany",
                    "away": "Spain",
                    "date": "2024-01-15",
                    "home_goals": 1,
                    "away_goals": 0,
                },
            ]
        )
        elo = _elo([{}])  # 1 row vs 2 matches → mismatch
        ab = _team_abilities("Brazil", "France", "Germany", "Spain")
        with pytest.raises(ValueError):
            FeatureBuilder(_CFG).build(m, elo, ab)

    def test_when_elo_has_more_rows_than_sorted_df_then_value_error_is_raised(self) -> None:
        m = _matches([{"home": "Brazil", "away": "Germany", "home_goals": 2, "away_goals": 1}])
        elo = _elo([{}, {}, {}])  # 3 rows vs 1 match → mismatch
        with pytest.raises(ValueError):
            FeatureBuilder(_CFG).build(m, elo, _BRA_GER)

    def test_when_elo_length_matches_sorted_df_length_then_no_error_is_raised(self) -> None:
        m = _matches(
            [
                {
                    "home": "Brazil",
                    "away": "France",
                    "date": "2024-01-01",
                    "home_goals": 2,
                    "away_goals": 1,
                },
                {
                    "home": "Brazil",
                    "away": "France",
                    "date": "2024-01-15",
                    "home_goals": 0,
                    "away_goals": 0,
                },
            ]
        )
        result = FeatureBuilder(_CFG).build(m, _elo([{}, {}]), _BRA_FRA)
        assert len(result) == 2

    def test_when_elo_result_has_mismatched_length_then_value_error_is_raised(self) -> None:
        """Alignment guard is enforced when EloResult.match_diffs length mismatches."""
        m = _matches(
            [
                {
                    "home": "Brazil",
                    "away": "France",
                    "date": "2024-01-01",
                    "home_goals": 2,
                    "away_goals": 1,
                },
                {
                    "home": "Brazil",
                    "away": "France",
                    "date": "2024-01-15",
                    "home_goals": 1,
                    "away_goals": 0,
                },
            ]
        )
        fake = _FakeEloResult(_elo([{}]))  # 1 row vs 2 matches
        with pytest.raises(ValueError):
            build_features(m, fake, _BRA_FRA)


# ---------------------------------------------------------------------------
# AC5 — End-to-end no-leakage + determinism
# ---------------------------------------------------------------------------


class TestEndToEndNoLeakage:
    """Match-2 features must use only match-1 information (no forward leakage)."""

    def test_when_first_match_has_no_prior_history_then_home_form_is_zero(self) -> None:
        m = _matches(
            [
                {
                    "home": "Brazil",
                    "away": "Germany",
                    "date": "2026-01-01",
                    "home_goals": 3,
                    "away_goals": 0,
                },
            ]
        )
        result = FeatureBuilder(_CFG).build(m, _elo([{}]), _BRA_GER)
        assert result.iloc[0]["home_form"] == pytest.approx(0.0)

    def test_when_match2_is_computed_then_home_form_equals_match1_win_points(self) -> None:
        """
        Two-match history for Brazil:
          match-1: Brazil beats Germany 3-0 on 2026-01-01 (Brazil earns 3 pts)
          match-2: Brazil vs Germany on 2026-01-10

        At match-2, Brazil's home_form must reflect match-1's 3-0 win.
        With a single prior played match, PPG = 3.0 regardless of decay weight.
        """
        m = _matches(
            [
                {
                    "home": "Brazil",
                    "away": "Germany",
                    "date": "2026-01-01",
                    "home_goals": 3,
                    "away_goals": 0,
                },
                {
                    "home": "Brazil",
                    "away": "Germany",
                    "date": "2026-01-10",
                    "home_goals": 1,
                    "away_goals": 2,
                },
            ]
        )
        result = FeatureBuilder(_CFG).build(m, _elo([{}, {}]), _BRA_GER)
        assert result.iloc[1]["home_form"] == pytest.approx(3.0)

    def test_when_match2_result_is_a_loss_then_match2_away_form_reflects_only_match1(
        self,
    ) -> None:
        """
        Germany won match-1 as away team (Brazil 3-0 → Germany 0 pts).
        At match-2, Germany's away_form must show 0.0 (a prior loss), NOT be
        influenced by match-2's result (Germany wins 2-1, which would give 3 pts).
        """
        m = _matches(
            [
                {
                    "home": "Brazil",
                    "away": "Germany",
                    "date": "2026-01-01",
                    "home_goals": 3,
                    "away_goals": 0,
                },
                {
                    "home": "Brazil",
                    "away": "Germany",
                    "date": "2026-01-10",
                    "home_goals": 1,
                    "away_goals": 2,
                },
            ]
        )
        result = FeatureBuilder(_CFG).build(m, _elo([{}, {}]), _BRA_GER)
        assert result.iloc[1]["away_form"] == pytest.approx(0.0)

    def test_when_match2_goal_diff_is_read_then_it_reflects_only_match1_result(self) -> None:
        m = _matches(
            [
                {
                    "home": "Brazil",
                    "away": "Germany",
                    "date": "2026-01-01",
                    "home_goals": 3,
                    "away_goals": 1,
                },
                {
                    "home": "Brazil",
                    "away": "Germany",
                    "date": "2026-01-10",
                    "home_goals": 0,
                    "away_goals": 0,
                },
            ]
        )
        result = FeatureBuilder(_CFG).build(m, _elo([{}, {}]), _BRA_GER)
        assert result.iloc[1]["home_goal_diff"] == pytest.approx(2.0)
        assert result.iloc[1]["away_goal_diff"] == pytest.approx(-2.0)

    def test_when_future_match_appended_then_first_match_home_form_is_unchanged(
        self,
    ) -> None:
        """Adding a later match must not alter an earlier row's form — no future bleed."""
        m_one = _matches(
            [
                {
                    "home": "Brazil",
                    "away": "Germany",
                    "date": "2026-01-01",
                    "home_goals": 3,
                    "away_goals": 0,
                }
            ]
        )
        result_one = FeatureBuilder(_CFG).build(m_one, _elo([{}]), _BRA_GER)
        form_standalone = result_one.iloc[0]["home_form"]

        m_two = _matches(
            [
                {
                    "home": "Brazil",
                    "away": "Germany",
                    "date": "2026-01-01",
                    "home_goals": 3,
                    "away_goals": 0,
                },
                {
                    "home": "Brazil",
                    "away": "Germany",
                    "date": "2026-01-15",
                    "home_goals": 2,
                    "away_goals": 2,
                },
            ]
        )
        result_two = FeatureBuilder(_CFG).build(m_two, _elo([{}, {}]), _BRA_GER)
        form_with_future = result_two.iloc[0]["home_form"]

        assert form_standalone == pytest.approx(form_with_future), (
            "A later match must not alter the form of the first match row"
        )

    def test_when_same_inputs_provided_twice_then_outputs_are_bitwise_identical(self) -> None:
        """build_features is deterministic: identical inputs → identical outputs."""
        m = _matches(
            [
                {
                    "home": "Brazil",
                    "away": "Germany",
                    "date": "2026-01-01",
                    "home_goals": 2,
                    "away_goals": 1,
                },
                {
                    "home": "Brazil",
                    "away": "Germany",
                    "date": "2026-01-15",
                    "home_goals": 1,
                    "away_goals": 0,
                },
            ]
        )
        result_a = build_features(m.copy(), _elo([{}, {}]), _BRA_GER)
        result_b = build_features(m.copy(), _elo([{}, {}]), _BRA_GER)
        pd.testing.assert_frame_equal(result_a, result_b)


# ---------------------------------------------------------------------------
# Property-based tests (Hypothesis)
# ---------------------------------------------------------------------------


@given(
    home_elo=st.floats(min_value=500.0, max_value=2500.0, allow_nan=False, allow_infinity=False),
    away_elo=st.floats(min_value=500.0, max_value=2500.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_when_any_valid_elo_pair_then_elo_diff_equals_home_minus_away(
    home_elo: float, away_elo: float
) -> None:
    """Invariant: elo_diff == home_elo − away_elo for all valid floating-point elo pairs."""
    m = _matches([{"home": "Brazil", "away": "Germany", "home_goals": 2, "away_goals": 1}])
    elo = _elo([{"home_elo": home_elo, "away_elo": away_elo}])
    result = FeatureBuilder(_CFG).build(m, elo, _BRA_GER)
    assert result["elo_diff"].iloc[0] == pytest.approx(home_elo - away_elo, rel=1e-9)


@given(n_future=st.integers(min_value=0, max_value=25))
@settings(max_examples=50)
def test_when_any_number_of_later_matches_present_then_first_match_home_form_is_zero(
    n_future: int,
) -> None:
    """Ordering invariant: first chronological match always has home_form == 0.0
    regardless of how many later matches are included in the same build call."""
    rows = [
        {
            "home": "Brazil",
            "away": "Germany",
            "date": "2020-01-01",
            "home_goals": 2,
            "away_goals": 1,
        }
    ]
    for i in range(n_future):
        day = 2 + i  # 2020-01-02 … 2020-01-27 — all valid January dates
        rows.append(
            {
                "home": "Brazil",
                "away": "Germany",
                "date": f"2020-01-{day:02d}",
                "home_goals": 1,
                "away_goals": 0,
            }
        )
    m = _matches(rows)
    result = FeatureBuilder(_CFG).build(m, _elo([{}] * len(rows)), _BRA_GER)
    assert result["home_form"].iloc[0] == pytest.approx(0.0)
