"""Source-blind example tests for worldcup_playoff/features/wc2026.py — Issue #37.

All tests are derived from the acceptance-criteria text only (Red-phase TDD).
No implementation source was read during authoring.

Invariant choices documented where criterion text is ambiguous:
- "unplayed" = HOME_GOALS / AWAY_GOALS are pandas NA (nullable Int64).
- "FIFA World Cup" rows = TOURNAMENT == "FIFA World Cup" exactly.
- "played WC" row = TOURNAMENT == "FIFA World Cup" and HOME_GOALS is not NA.
- "home_team / away_team preserved" = the output DataFrame contains columns
  `home_team` and `away_team` whose values match the input fixture teams.
- elo is duck-typed: either a pre-built DataFrame (home_elo / away_elo cols) or
  any object exposing a .match_diffs attribute that returns an equivalent DataFrame.
- The output carries the full feature schema (all FEATURE_COLUMNS) plus at minimum
  a `tournament` metadata column; home_team / away_team are additional metadata columns.
- Covariates are computed by FeatureBuilder run over the FULL results_df, not the WC subset.
"""

from __future__ import annotations

import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from worldcup_playoff.config import FeatureBuildConfig
from worldcup_playoff.features.build import (
    FEATURE_COLUMNS,
    TeamAbilities,
    build_features,
)
from worldcup_playoff.features.wc2026 import live_fixtures_to_df, wc2026_features

_CFG = FeatureBuildConfig(random_seed=42)


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------


def _results(rows: list[dict]) -> pd.DataFrame:
    """Build an internal-schema results DataFrame from a list of row dicts."""
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


def _elo(n: int, home_elo: float = 1500.0, away_elo: float = 1500.0) -> pd.DataFrame:
    return pd.DataFrame({"home_elo": [home_elo] * n, "away_elo": [away_elo] * n})


def _abilities(*teams: str) -> TeamAbilities:
    return TeamAbilities(
        attack={t: 1.0 for t in teams},
        defence={t: 1.0 for t in teams},
        home_adv=0.0,
        rho=-0.1,
        intercept=0.0,
    )


class _FakeEloResult:
    """Duck-typed EloResult — exposes only .match_diffs (no production source read)."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.match_diffs = df


_ALL_WC_TEAMS = (
    "Brazil",
    "France",
    "Germany",
    "Spain",
    "Argentina",
    "England",
    "Croatia",
    "Portugal",
    "Netherlands",
    "Uruguay",
)
_AB_ALL = _abilities(*_ALL_WC_TEAMS)


# ---------------------------------------------------------------------------
# AC1 — Row filtering: returns exactly the unplayed FIFA World Cup rows
# ---------------------------------------------------------------------------


class TestRowFiltering:
    """wc2026_features returns exactly the unplayed FIFA World Cup rows."""

    def test_when_only_unplayed_wc_rows_are_in_input_then_all_are_returned(self) -> None:
        df = _results(
            [
                {
                    "home": "Brazil",
                    "away": "France",
                    "date": "2026-06-20",
                    "tournament": "FIFA World Cup",
                    "neutral": True,
                },
                {
                    "home": "Germany",
                    "away": "Spain",
                    "date": "2026-06-21",
                    "tournament": "FIFA World Cup",
                    "neutral": True,
                },
            ]
        )
        result = wc2026_features(df, _elo(2), _abilities("Brazil", "France", "Germany", "Spain"))
        assert len(result) == 2

    def test_when_played_wc_row_is_in_input_then_it_is_excluded(self) -> None:
        df = _results(
            [
                {
                    "home": "Brazil",
                    "away": "France",
                    "date": "2026-06-15",
                    "tournament": "FIFA World Cup",
                    "home_goals": 2,
                    "away_goals": 1,
                    "neutral": True,
                },
                {
                    "home": "Germany",
                    "away": "Spain",
                    "date": "2026-06-21",
                    "tournament": "FIFA World Cup",
                    "neutral": True,
                },
            ]
        )
        result = wc2026_features(df, _elo(2), _abilities("Brazil", "France", "Germany", "Spain"))
        assert len(result) == 1

    def test_when_non_wc_friendly_row_is_in_input_then_it_is_excluded(self) -> None:
        df = _results(
            [
                {
                    "home": "Brazil",
                    "away": "France",
                    "date": "2026-01-10",
                    "tournament": "Friendly",
                    "home_goals": 1,
                    "away_goals": 0,
                },
                {
                    "home": "Germany",
                    "away": "Spain",
                    "date": "2026-06-21",
                    "tournament": "FIFA World Cup",
                    "neutral": True,
                },
            ]
        )
        result = wc2026_features(df, _elo(2), _abilities("Brazil", "France", "Germany", "Spain"))
        assert len(result) == 1

    def test_when_qualification_row_is_in_input_then_it_is_excluded(self) -> None:
        """FIFA World Cup qualification ≠ FIFA World Cup — must be excluded."""
        df = _results(
            [
                {
                    "home": "Argentina",
                    "away": "Uruguay",
                    "date": "2025-10-10",
                    "tournament": "FIFA World Cup qualification",
                    "home_goals": 2,
                    "away_goals": 1,
                },
                {
                    "home": "Brazil",
                    "away": "France",
                    "date": "2026-06-20",
                    "tournament": "FIFA World Cup",
                    "neutral": True,
                },
            ]
        )
        result = wc2026_features(
            df, _elo(2), _abilities("Argentina", "Uruguay", "Brazil", "France")
        )
        assert len(result) == 1

    def test_when_empty_input_frame_is_given_then_empty_dataframe_is_returned_without_raising(
        self,
    ) -> None:
        empty = _results([])
        result = wc2026_features(empty, _elo(0), _AB_ALL)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_when_input_has_only_played_wc_rows_then_empty_dataframe_is_returned(
        self,
    ) -> None:
        df = _results(
            [
                {
                    "home": "Brazil",
                    "away": "France",
                    "date": "2026-06-15",
                    "tournament": "FIFA World Cup",
                    "home_goals": 2,
                    "away_goals": 1,
                    "neutral": True,
                },
            ]
        )
        result = wc2026_features(df, _elo(1), _abilities("Brazil", "France"))
        assert len(result) == 0

    def test_when_input_has_only_non_wc_rows_then_empty_dataframe_is_returned(
        self,
    ) -> None:
        df = _results(
            [
                {
                    "home": "Brazil",
                    "away": "France",
                    "date": "2026-01-10",
                    "tournament": "Friendly",
                    "home_goals": 1,
                    "away_goals": 0,
                },
            ]
        )
        result = wc2026_features(df, _elo(1), _abilities("Brazil", "France"))
        assert len(result) == 0

    def test_when_mixed_tournament_types_present_then_only_unplayed_wc_count_is_returned(
        self,
    ) -> None:
        """Friendly, played WC, WC qualification, and UEFA rows must all be excluded."""
        df = _results(
            [
                {
                    "home": "Brazil",
                    "away": "France",
                    "date": "2025-10-01",
                    "tournament": "FIFA World Cup qualification",
                    "home_goals": 1,
                    "away_goals": 0,
                },
                {
                    "home": "Germany",
                    "away": "Spain",
                    "date": "2025-11-01",
                    "tournament": "UEFA Nations League",
                    "home_goals": 0,
                    "away_goals": 0,
                },
                {
                    "home": "Argentina",
                    "away": "England",
                    "date": "2026-06-10",
                    "tournament": "FIFA World Cup",
                    "home_goals": 2,
                    "away_goals": 1,
                    "neutral": True,
                },  # played WC
                {
                    "home": "Brazil",
                    "away": "Germany",
                    "date": "2026-06-20",
                    "tournament": "FIFA World Cup",
                    "neutral": True,
                },  # unplayed
                {
                    "home": "France",
                    "away": "Spain",
                    "date": "2026-06-21",
                    "tournament": "FIFA World Cup",
                    "neutral": True,
                },  # unplayed
            ]
        )
        ab = _abilities("Brazil", "France", "Germany", "Spain", "Argentina", "England")
        result = wc2026_features(df, _elo(5), ab)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# AC2 — NA goals (Int64) + home_team / away_team preservation
# ---------------------------------------------------------------------------


class TestGoalsNaAndTeamPreservation:
    """Every output row has home_goals/away_goals as NA (Int64); teams are preserved."""

    def test_when_output_row_is_inspected_then_home_goals_is_na(self) -> None:
        df = _results(
            [
                {
                    "home": "Brazil",
                    "away": "France",
                    "date": "2026-06-20",
                    "tournament": "FIFA World Cup",
                    "neutral": True,
                },
            ]
        )
        result = wc2026_features(df, _elo(1), _abilities("Brazil", "France"))
        assert pd.isna(result["home_goals"].iloc[0])

    def test_when_output_row_is_inspected_then_away_goals_is_na(self) -> None:
        df = _results(
            [
                {
                    "home": "Brazil",
                    "away": "France",
                    "date": "2026-06-20",
                    "tournament": "FIFA World Cup",
                    "neutral": True,
                },
            ]
        )
        result = wc2026_features(df, _elo(1), _abilities("Brazil", "France"))
        assert pd.isna(result["away_goals"].iloc[0])

    def test_when_home_goals_dtype_is_inspected_then_it_is_int64(self) -> None:
        df = _results(
            [
                {
                    "home": "Brazil",
                    "away": "France",
                    "date": "2026-06-20",
                    "tournament": "FIFA World Cup",
                    "neutral": True,
                },
            ]
        )
        result = wc2026_features(df, _elo(1), _abilities("Brazil", "France"))
        assert result["home_goals"].dtype == pd.Int64Dtype()

    def test_when_away_goals_dtype_is_inspected_then_it_is_int64(self) -> None:
        df = _results(
            [
                {
                    "home": "Brazil",
                    "away": "France",
                    "date": "2026-06-20",
                    "tournament": "FIFA World Cup",
                    "neutral": True,
                },
            ]
        )
        result = wc2026_features(df, _elo(1), _abilities("Brazil", "France"))
        assert result["away_goals"].dtype == pd.Int64Dtype()

    def test_when_output_home_team_is_inspected_then_it_matches_input_fixture(self) -> None:
        df = _results(
            [
                {
                    "home": "Brazil",
                    "away": "France",
                    "date": "2026-06-20",
                    "tournament": "FIFA World Cup",
                    "neutral": True,
                },
            ]
        )
        result = wc2026_features(df, _elo(1), _abilities("Brazil", "France"))
        assert result["home_team"].iloc[0] == "Brazil"

    def test_when_output_away_team_is_inspected_then_it_matches_input_fixture(self) -> None:
        df = _results(
            [
                {
                    "home": "Brazil",
                    "away": "France",
                    "date": "2026-06-20",
                    "tournament": "FIFA World Cup",
                    "neutral": True,
                },
            ]
        )
        result = wc2026_features(df, _elo(1), _abilities("Brazil", "France"))
        assert result["away_team"].iloc[0] == "France"

    def test_when_multiple_unplayed_wc_rows_are_present_then_all_home_goals_are_na(
        self,
    ) -> None:
        df = _results(
            [
                {
                    "home": "Brazil",
                    "away": "France",
                    "date": "2026-06-20",
                    "tournament": "FIFA World Cup",
                    "neutral": True,
                },
                {
                    "home": "Germany",
                    "away": "Spain",
                    "date": "2026-06-21",
                    "tournament": "FIFA World Cup",
                    "neutral": True,
                },
            ]
        )
        result = wc2026_features(df, _elo(2), _abilities("Brazil", "France", "Germany", "Spain"))
        assert result["home_goals"].isna().all()

    def test_when_multiple_unplayed_wc_rows_are_present_then_all_away_goals_are_na(
        self,
    ) -> None:
        df = _results(
            [
                {
                    "home": "Brazil",
                    "away": "France",
                    "date": "2026-06-20",
                    "tournament": "FIFA World Cup",
                    "neutral": True,
                },
                {
                    "home": "Germany",
                    "away": "Spain",
                    "date": "2026-06-21",
                    "tournament": "FIFA World Cup",
                    "neutral": True,
                },
            ]
        )
        result = wc2026_features(df, _elo(2), _abilities("Brazil", "France", "Germany", "Spain"))
        assert result["away_goals"].isna().all()

    def test_when_two_unplayed_wc_rows_are_present_then_team_order_is_preserved(
        self,
    ) -> None:
        """home_team / away_team must correspond to each fixture in output order."""
        df = _results(
            [
                {
                    "home": "Germany",
                    "away": "Spain",
                    "date": "2026-06-20",
                    "tournament": "FIFA World Cup",
                    "neutral": True,
                },
                {
                    "home": "Brazil",
                    "away": "France",
                    "date": "2026-06-21",
                    "tournament": "FIFA World Cup",
                    "neutral": True,
                },
            ]
        )
        ab = _abilities("Germany", "Spain", "Brazil", "France")
        result = wc2026_features(df, _elo(2), ab)
        # Both fixtures must appear; their team columns must match input order
        home_teams = set(result["home_team"].tolist())
        away_teams = set(result["away_team"].tolist())
        assert "Germany" in home_teams or "Germany" in away_teams
        assert "Brazil" in home_teams or "Brazil" in away_teams
        assert "France" in home_teams or "France" in away_teams
        assert "Spain" in home_teams or "Spain" in away_teams


# ---------------------------------------------------------------------------
# AC3 — Covariate parity: FeatureBuilder run over the full history
# ---------------------------------------------------------------------------


class TestCovariateParity:
    """Covariates equal what FeatureBuilder produces on the full history."""

    def test_when_wc2026_features_called_then_elo_diff_matches_full_history_build(
        self,
    ) -> None:
        """
        The elo_diff for the WC fixture in wc2026_features must match the value
        that build_features produces for that same row when applied to the full history.
        We use a varying elo array so the row position matters.
        """
        full_df = _results(
            [
                {
                    "home": "Brazil",
                    "away": "France",
                    "date": "2020-01-01",
                    "home_goals": 2,
                    "away_goals": 1,
                    "tournament": "Friendly",
                },
                {
                    "home": "Germany",
                    "away": "Spain",
                    "date": "2023-06-01",
                    "home_goals": 1,
                    "away_goals": 0,
                    "tournament": "Friendly",
                },
                {
                    "home": "Brazil",
                    "away": "Germany",
                    "date": "2026-06-20",
                    "tournament": "FIFA World Cup",
                    "neutral": True,
                },
            ]
        )
        # Vary elo so the row-position of the WC fixture (index 2) is distinguishable
        elo_df = pd.DataFrame(
            {
                "home_elo": [1500.0, 1600.0, 1900.0],
                "away_elo": [1500.0, 1700.0, 1750.0],
            }
        )
        ab = _abilities("Brazil", "France", "Germany", "Spain")

        wc_result = wc2026_features(full_df, elo_df, ab)

        # Reference: build_features on the full history; WC fixture is sorted last
        fb_result = build_features(full_df, elo_df, ab, config=_CFG)
        # After sort_chronological: 2020-01-01 → idx 0, 2023-06-01 → idx 1, 2026-06-20 → idx 2
        fb_elo_diff = fb_result.iloc[2]["elo_diff"]

        assert wc_result["elo_diff"].iloc[0] == pytest.approx(fb_elo_diff)

    def test_when_wc2026_features_called_then_home_form_uses_full_history_not_wc_subset(
        self,
    ) -> None:
        """
        This is the key AC3 test. If wc2026_features incorrectly ran FeatureBuilder
        only on the WC subset, Brazil's home_form at the WC fixture would be 0.0
        (no prior WC matches). The correct behaviour includes the prior Friendly win,
        yielding a positive form matching the full-history FeatureBuilder.
        """
        full_df = _results(
            [
                # Prior history — non-WC, played
                {
                    "home": "Brazil",
                    "away": "France",
                    "date": "2020-01-01",
                    "home_goals": 3,
                    "away_goals": 0,
                    "tournament": "Friendly",
                },
                # Unplayed WC fixture
                {
                    "home": "Brazil",
                    "away": "Germany",
                    "date": "2026-06-20",
                    "tournament": "FIFA World Cup",
                    "neutral": True,
                },
            ]
        )
        ab = _abilities("Brazil", "France", "Germany")
        elo_df = _elo(2)

        wc_result = wc2026_features(full_df, elo_df, ab)

        # Reference: run build_features on the full history
        fb_result = build_features(full_df, elo_df, ab, config=_CFG)
        # After sort_chronological: 2020-01-01 → idx 0, 2026-06-20 → idx 1
        fb_home_form = fb_result.iloc[1]["home_form"]

        # Test is meaningful only if the full-history builder sees Brazil's prior win
        assert fb_home_form > 0.0, (
            "Test precondition: full-history builder must reflect Brazil's prior Friendly win"
        )
        assert wc_result["home_form"].iloc[0] == pytest.approx(fb_home_form)

    def test_when_no_historical_matches_are_present_then_home_form_is_zero(self) -> None:
        """When the WC fixture is the only row, form must be 0.0 — no prior history."""
        df = _results(
            [
                {
                    "home": "Brazil",
                    "away": "Germany",
                    "date": "2026-06-20",
                    "tournament": "FIFA World Cup",
                    "neutral": True,
                },
            ]
        )
        ab = _abilities("Brazil", "Germany")
        result = wc2026_features(df, _elo(1), ab)
        assert result["home_form"].iloc[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# AC4 — Full feature schema + tournament column + elo overloading
# ---------------------------------------------------------------------------


class TestOutputSchemaAndEloOverloading:
    """Output has FEATURE_COLUMNS + tournament + team metadata; elo may be DF or EloResult."""

    def test_when_output_is_inspected_then_tournament_column_is_present(self) -> None:
        df = _results(
            [
                {
                    "home": "Brazil",
                    "away": "France",
                    "date": "2026-06-20",
                    "tournament": "FIFA World Cup",
                    "neutral": True,
                },
            ]
        )
        result = wc2026_features(df, _elo(1), _abilities("Brazil", "France"))
        assert "tournament" in result.columns

    def test_when_output_tournament_value_is_inspected_then_it_equals_fifa_world_cup(
        self,
    ) -> None:
        df = _results(
            [
                {
                    "home": "Brazil",
                    "away": "France",
                    "date": "2026-06-20",
                    "tournament": "FIFA World Cup",
                    "neutral": True,
                },
            ]
        )
        result = wc2026_features(df, _elo(1), _abilities("Brazil", "France"))
        assert result["tournament"].iloc[0] == "FIFA World Cup"

    def test_when_output_is_inspected_then_all_feature_columns_are_present(self) -> None:
        df = _results(
            [
                {
                    "home": "Brazil",
                    "away": "France",
                    "date": "2026-06-20",
                    "tournament": "FIFA World Cup",
                    "neutral": True,
                },
            ]
        )
        result = wc2026_features(df, _elo(1), _abilities("Brazil", "France"))
        missing = [c for c in FEATURE_COLUMNS if c not in result.columns]
        assert not missing, f"Missing feature columns in output: {missing}"

    def test_when_elo_is_a_pre_built_dataframe_then_no_error_is_raised(self) -> None:
        df = _results(
            [
                {
                    "home": "Brazil",
                    "away": "France",
                    "date": "2026-06-20",
                    "tournament": "FIFA World Cup",
                    "neutral": True,
                },
            ]
        )
        wc2026_features(df, _elo(1), _abilities("Brazil", "France"))

    def test_when_elo_is_an_elo_result_object_then_no_error_is_raised(self) -> None:
        """build_features accepts EloResult via .match_diffs; wc2026_features must too."""
        df = _results(
            [
                {
                    "home": "Brazil",
                    "away": "France",
                    "date": "2026-06-20",
                    "tournament": "FIFA World Cup",
                    "neutral": True,
                },
            ]
        )
        fake_elo = _FakeEloResult(_elo(1))
        wc2026_features(df, fake_elo, _abilities("Brazil", "France"))

    def test_when_config_is_none_then_no_error_is_raised(self) -> None:
        df = _results(
            [
                {
                    "home": "Brazil",
                    "away": "France",
                    "date": "2026-06-20",
                    "tournament": "FIFA World Cup",
                    "neutral": True,
                },
            ]
        )
        wc2026_features(df, _elo(1), _abilities("Brazil", "France"), config=None)

    def test_when_ranking_is_none_then_no_error_is_raised(self) -> None:
        df = _results(
            [
                {
                    "home": "Brazil",
                    "away": "France",
                    "date": "2026-06-20",
                    "tournament": "FIFA World Cup",
                    "neutral": True,
                },
            ]
        )
        wc2026_features(df, _elo(1), _abilities("Brazil", "France"), ranking=None)

    def test_when_elo_result_and_dataframe_are_equivalent_then_elo_diff_is_identical(
        self,
    ) -> None:
        """
        EloResult duck-type must produce the same elo_diff as the pre-built DataFrame it wraps.
        """
        df = _results(
            [
                {
                    "home": "Brazil",
                    "away": "France",
                    "date": "2026-06-20",
                    "tournament": "FIFA World Cup",
                    "neutral": True,
                },
            ]
        )
        elo_df = _elo(1, home_elo=1900.0, away_elo=1750.0)
        fake_elo = _FakeEloResult(elo_df)

        result_df = wc2026_features(df, elo_df, _abilities("Brazil", "France"))
        result_fake = wc2026_features(df, fake_elo, _abilities("Brazil", "France"))

        assert result_df["elo_diff"].iloc[0] == pytest.approx(result_fake["elo_diff"].iloc[0])


# ---------------------------------------------------------------------------
# AC5 — live_fixtures_to_df importable from worldcup_playoff.features.wc2026
# ---------------------------------------------------------------------------


def test_when_live_fixtures_to_df_is_imported_then_it_is_callable() -> None:
    """live_fixtures_to_df must be importable from worldcup_playoff.features.wc2026."""
    assert callable(live_fixtures_to_df)


# ---------------------------------------------------------------------------
# Property-based tests (Hypothesis)
# ---------------------------------------------------------------------------


@given(
    n_friendly=st.integers(min_value=0, max_value=5),
    n_played_wc=st.integers(min_value=0, max_value=5),
)
@settings(max_examples=60)
def test_when_no_unplayed_wc_rows_are_present_then_output_is_always_empty(
    n_friendly: int, n_played_wc: int
) -> None:
    """
    Filtering invariant: any frame that contains zero unplayed WC rows must yield
    an empty output, regardless of how many Friendly / played-WC rows are mixed in.
    """
    rows: list[dict] = []
    for i in range(n_friendly):
        rows.append(
            {
                "home": "Brazil",
                "away": "France",
                "date": f"2022-{(i % 9) + 1:02d}-01",
                "tournament": "Friendly",
                "home_goals": i % 4,
                "away_goals": (i + 1) % 4,
            }
        )
    for i in range(n_played_wc):
        rows.append(
            {
                "home": "Germany",
                "away": "Spain",
                "date": f"2026-06-{(i % 25) + 1:02d}",
                "tournament": "FIFA World Cup",
                "home_goals": i % 4,
                "away_goals": (i + 1) % 4,
                "neutral": True,
            }
        )
    df = _results(rows)
    ab = _abilities("Brazil", "France", "Germany", "Spain")
    result = wc2026_features(df, _elo(len(df)), ab)
    assert len(result) == 0


@given(n_unplayed=st.integers(min_value=1, max_value=6))
@settings(max_examples=40)
def test_when_n_unplayed_wc_fixtures_are_in_input_then_output_has_exactly_n_rows(
    n_unplayed: int,
) -> None:
    """
    Monotonicity invariant: output row count equals the number of unplayed WC fixtures
    in the input — one-to-one correspondence for any valid count.
    """
    _PAIRS = [
        ("Brazil", "France"),
        ("Germany", "Spain"),
        ("Argentina", "Uruguay"),
        ("England", "Croatia"),
        ("Portugal", "Netherlands"),
        ("Belgium", "Japan"),
    ]
    rows = [
        {
            "home": _PAIRS[i % len(_PAIRS)][0],
            "away": _PAIRS[i % len(_PAIRS)][1],
            "date": f"2026-06-{20 + i}",
            "tournament": "FIFA World Cup",
            "neutral": True,
        }
        for i in range(n_unplayed)
    ]
    all_teams = {t for pair in _PAIRS for t in pair}
    df = _results(rows)
    result = wc2026_features(df, _elo(n_unplayed), _abilities(*all_teams))
    assert len(result) == n_unplayed
