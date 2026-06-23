"""
Source-blind example tests for issue #10 criterion 1:

    Given an in-memory martj42 frame containing unplayed FIFA World Cup rows,
    wc2026_features(...) returns those rows with the full feature schema and
    home_goals/away_goals as <NA>.

Tests are derived solely from the acceptance criteria and requirements.md.
No implementation source was read; these tests describe the contract the
implementation must satisfy (Red phase of TDD).

Choices recorded where the spec is silent:
  - wc2026_features resides in worldcup_playoff.features.wc2026.
  - Its signature mirrors FeatureBuilder.build(): (results_df, elo_df,
    abilities [, ranking_df, cfg]) — the same pre-computed inputs that the
    full-history builder already accepts (issue #9).
  - Input DataFrame uses the uppercase martj42 internal schema
    (DATE, HOME_TEAM, AWAY_TEAM, HOME_GOALS, AWAY_GOALS, TOURNAMENT, NEUTRAL)
    consistent with test_feature_builder.py.
  - Feature output uses lowercase column names (home_goals, away_goals,
    home_team, away_team, tournament) as observed in FeatureBuilder.build()
    output in test_feature_builder.py.
  - Unplayed rows are identified by HOME_GOALS.isna() in the results frame.

Criteria skipped (NOT VERIFIABLE per oracle):
  - WC2026 fixture features equal what the full-history builder produces for
    the same rows (consistency / single covariate code path).
  - No-key path produces a usable WC2026 frame without any API key.
  - Deterministic given the seed.
  - All tests pass / SOLID quality gates.
"""

from __future__ import annotations

import pandas as pd
from hypothesis import given, settings, strategies as st

from worldcup_playoff.config import FeatureBuildConfig
from worldcup_playoff.features.build import TeamAbilities

# ---------------------------------------------------------------------------
# Shared helpers — derived from data contracts in requirements.md
# ---------------------------------------------------------------------------

_CFG = FeatureBuildConfig(random_seed=42)


def _matches(rows: list[dict]) -> pd.DataFrame:
    """Build a results DataFrame in the martj42 uppercase internal schema."""
    return pd.DataFrame(
        {
            "DATE": pd.to_datetime([r.get("date", "2026-06-20") for r in rows]),
            "HOME_TEAM": [r["home"] for r in rows],
            "AWAY_TEAM": [r["away"] for r in rows],
            "HOME_GOALS": pd.array([r.get("home_goals", pd.NA) for r in rows], dtype="Int64"),
            "AWAY_GOALS": pd.array([r.get("away_goals", pd.NA) for r in rows], dtype="Int64"),
            "TOURNAMENT": [r.get("tournament", "Friendly") for r in rows],
            "NEUTRAL": [r.get("neutral", True) for r in rows],
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


def _abilities(*team_names: str) -> TeamAbilities:
    """Build a uniform TeamAbilities for the named teams."""
    return TeamAbilities(
        attack={t: 1.3 for t in team_names},
        defence={t: 0.9 for t in team_names},
        home_adv=0.0,
        rho=-0.1,
        intercept=0.0,
    )


def _mixed_frame_and_elo():
    """Full frame: historical rows + played WC rows + two unplayed WC fixtures."""
    rows = [
        {
            "home": "Argentina",
            "away": "El Salvador",
            "tournament": "Friendly",
            "home_goals": 3,
            "away_goals": 0,
            "date": "2024-03-20",
            "neutral": False,
        },
        {
            "home": "Brazil",
            "away": "Mexico",
            "tournament": "Friendly",
            "home_goals": 2,
            "away_goals": 1,
            "date": "2024-06-10",
            "neutral": False,
        },
        {
            "home": "Mexico",
            "away": "Canada",
            "tournament": "FIFA World Cup",
            "home_goals": 2,
            "away_goals": 0,
            "date": "2026-06-11",
        },
        # unplayed WC2026 fixtures — HOME_GOALS/AWAY_GOALS default to pd.NA
        {
            "home": "Argentina",
            "away": "France",
            "tournament": "FIFA World Cup",
            "date": "2026-06-15",
        },
        {"home": "Brazil", "away": "Spain", "tournament": "FIFA World Cup", "date": "2026-06-16"},
    ]
    m = _matches(rows)
    elo = _elo([{}, {}, {}, {}, {}])
    return m, elo


_ALL_TEAMS = ("Argentina", "El Salvador", "Brazil", "Mexico", "Canada", "France", "Spain")


# ---------------------------------------------------------------------------
# Criterion 1a — row filtering: only unplayed WC fixtures appear in output
# ---------------------------------------------------------------------------


class TestWc2026FeatureRowFiltering:
    """wc2026_features must return exactly the unplayed FIFA World Cup rows."""

    def test_when_mixed_frame_given_then_only_unplayed_wc_rows_are_returned(self) -> None:
        from worldcup_playoff.features.wc2026 import wc2026_features

        m, elo = _mixed_frame_and_elo()
        result = wc2026_features(m, elo, _abilities(*_ALL_TEAMS))

        assert len(result) == 2

    def test_when_wc2026_features_called_then_non_wc_rows_are_absent(self) -> None:
        from worldcup_playoff.features.wc2026 import wc2026_features

        m, elo = _mixed_frame_and_elo()
        result = wc2026_features(m, elo, _abilities(*_ALL_TEAMS))

        assert "Friendly" not in result["tournament"].values

    def test_when_wc2026_features_called_then_played_wc_rows_are_absent(self) -> None:
        """Played WC matches (goals not NA) must be excluded from the output."""
        from worldcup_playoff.features.wc2026 import wc2026_features

        rows = [
            {
                "home": "Mexico",
                "away": "Canada",
                "tournament": "FIFA World Cup",
                "home_goals": 2,
                "away_goals": 0,
                "date": "2026-06-11",
            },
            {
                "home": "Brazil",
                "away": "Spain",
                "tournament": "FIFA World Cup",
                "date": "2026-06-16",
            },
        ]
        m = _matches(rows)
        elo = _elo([{}, {}])
        result = wc2026_features(m, elo, _abilities("Mexico", "Canada", "Brazil", "Spain"))

        assert len(result) == 1
        assert result.iloc[0]["home_team"] == "Brazil"

    def test_when_no_wc_rows_in_frame_then_result_is_empty_dataframe(self) -> None:
        from worldcup_playoff.features.wc2026 import wc2026_features

        m = _matches(
            [
                {
                    "home": "Germany",
                    "away": "England",
                    "tournament": "Friendly",
                    "home_goals": 1,
                    "away_goals": 0,
                    "neutral": False,
                }
            ]
        )
        result = wc2026_features(m, _elo([{}]), _abilities("Germany", "England"))

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_when_only_played_wc_rows_then_result_is_empty_dataframe(self) -> None:
        """A frame where all WC rows are already played must produce an empty output."""
        from worldcup_playoff.features.wc2026 import wc2026_features

        m = _matches(
            [
                {
                    "home": "France",
                    "away": "Germany",
                    "tournament": "FIFA World Cup",
                    "home_goals": 1,
                    "away_goals": 0,
                    "date": "2026-06-14",
                }
            ]
        )
        result = wc2026_features(m, _elo([{}]), _abilities("France", "Germany"))

        assert len(result) == 0


# ---------------------------------------------------------------------------
# Criterion 1b — home_goals / away_goals must be <NA>
# ---------------------------------------------------------------------------


class TestWc2026FeatureNaGoals:
    """home_goals and away_goals must be <NA> in the output for every unplayed fixture."""

    def test_when_wc2026_features_called_then_home_goals_are_na(self) -> None:
        from worldcup_playoff.features.wc2026 import wc2026_features

        m, elo = _mixed_frame_and_elo()
        result = wc2026_features(m, elo, _abilities(*_ALL_TEAMS))

        assert result["home_goals"].isna().all()

    def test_when_wc2026_features_called_then_away_goals_are_na(self) -> None:
        from worldcup_playoff.features.wc2026 import wc2026_features

        m, elo = _mixed_frame_and_elo()
        result = wc2026_features(m, elo, _abilities(*_ALL_TEAMS))

        assert result["away_goals"].isna().all()

    def test_when_single_unplayed_wc_row_then_both_goals_are_na(self) -> None:
        from worldcup_playoff.features.wc2026 import wc2026_features

        m = _matches(
            [
                {
                    "home": "Germany",
                    "away": "Japan",
                    "tournament": "FIFA World Cup",
                    "date": "2026-06-17",
                }
            ]
        )
        result = wc2026_features(m, _elo([{}]), _abilities("Germany", "Japan"))

        assert len(result) == 1
        assert pd.isna(result.iloc[0]["home_goals"])
        assert pd.isna(result.iloc[0]["away_goals"])


# ---------------------------------------------------------------------------
# Criterion 1c — team names are preserved in the output
# ---------------------------------------------------------------------------


class TestWc2026FeatureTeamPreservation:
    """home_team / away_team values from the unplayed WC rows must appear in the output."""

    def test_when_wc2026_features_called_then_home_teams_are_preserved(self) -> None:
        from worldcup_playoff.features.wc2026 import wc2026_features

        m, elo = _mixed_frame_and_elo()
        result = wc2026_features(m, elo, _abilities(*_ALL_TEAMS))

        assert "Argentina" in result["home_team"].values
        assert "Brazil" in result["home_team"].values

    def test_when_wc2026_features_called_then_away_teams_are_preserved(self) -> None:
        from worldcup_playoff.features.wc2026 import wc2026_features

        m, elo = _mixed_frame_and_elo()
        result = wc2026_features(m, elo, _abilities(*_ALL_TEAMS))

        assert "France" in result["away_team"].values
        assert "Spain" in result["away_team"].values

    def test_when_wc2026_features_called_then_result_is_a_dataframe(self) -> None:
        from worldcup_playoff.features.wc2026 import wc2026_features

        m, elo = _mixed_frame_and_elo()
        result = wc2026_features(m, elo, _abilities(*_ALL_TEAMS))

        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# Property-based test — invariant from criterion 1
# ---------------------------------------------------------------------------


@given(n=st.integers(min_value=1, max_value=6))
@settings(max_examples=20)
def test_when_any_count_of_unplayed_wc_rows_then_goals_are_always_na(n: int) -> None:
    """
    Invariant: for ANY number of unplayed FIFA World Cup rows passed to
    wc2026_features, every row in the output has home_goals and away_goals
    equal to <NA>.

    Derived from criterion: 'returns those rows … with home_goals/away_goals as <NA>.'
    """
    from worldcup_playoff.features.wc2026 import wc2026_features

    team_pairs = [(f"Home{i}", f"Away{i}") for i in range(n)]
    all_teams = [t for pair in team_pairs for t in pair]

    rows = [
        {
            "home": home,
            "away": away,
            "tournament": "FIFA World Cup",
            "date": f"2026-07-{i + 1:02d}",
        }
        for i, (home, away) in enumerate(team_pairs)
    ]
    m = _matches(rows)
    elo = _elo([{}] * n)
    result = wc2026_features(m, elo, _abilities(*all_teams))

    assert result["home_goals"].isna().all(), "home_goals must be NA for all unplayed WC fixtures"
    assert result["away_goals"].isna().all(), "away_goals must be NA for all unplayed WC fixtures"
