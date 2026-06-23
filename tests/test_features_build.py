"""Supplemental tests for worldcup_playoff.features.build (Issue #9).

Covers acceptance criteria that the R4 oracle marked NOT VERIFIABLE but that
are concretely checkable once the implementation exists:

  2  Football-only allow-list: exact output column set, no forbidden columns.
  5  End-to-end no-leakage: match-2 features use only match-1 information.
"""

from __future__ import annotations

import pandas as pd
import pytest

from worldcup_playoff.config import FeatureBuildConfig
from worldcup_playoff.features.build import FEATURE_COLUMNS, FeatureBuilder, TeamAbilities

_CFG = FeatureBuildConfig(random_seed=42)

_FORBIDDEN_SUBSTRINGS = ("gdp", "market_value", "transfer", "odds", "bookie", "wage")


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


_BRA_GER = TeamAbilities(
    attack={"Brazil": 1.5, "Germany": 1.3},
    defence={"Brazil": 0.8, "Germany": 0.9},
    home_adv=0.0,
    rho=-0.1,
    intercept=0.0,
)


# ---------------------------------------------------------------------------
# Criterion 2 — Football-only allow-list
# ---------------------------------------------------------------------------


class TestFootballOnlyAllowList:
    """The exact output column set must be FEATURE_COLUMNS — no forbidden fields."""

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


# ---------------------------------------------------------------------------
# Criterion 5 — End-to-end no-leakage
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
        # No prior matches → neutral default form
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
        # match-2 form for Brazil = 3.0 (single prior win; decay cancels in PPG formula)
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
        # Germany's form at match-2 is based on match-1 only (a 3-0 loss → 0 pts)
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
        # Brazil GD at match-2: only match-1 (3-1 → +2)
        assert result.iloc[1]["home_goal_diff"] == pytest.approx(2.0)
        # Germany GD at match-2: only match-1 (1-3 → -2)
        assert result.iloc[1]["away_goal_diff"] == pytest.approx(-2.0)
