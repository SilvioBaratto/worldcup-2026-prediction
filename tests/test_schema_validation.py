"""Tests for DataFrame schema validation — train_data.csv column contract."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from worldcup_playoff.data.loader import validate_matches_df, validate_teams_df


# ---------------------------------------------------------------------------
# Expected column schemas
# ---------------------------------------------------------------------------

_EXPECTED_MATCHES_COLUMNS = {
    "MATCH_ID",
    "DATE",
    "HOME_TEAM",
    "AWAY_TEAM",
    "HOME_GOALS",
    "AWAY_GOALS",
    "COMPETITION",
    "SEASON",
}

_EXPECTED_TRAIN_DATA_COLUMNS = [
    "HOME_TEAM",
    "AWAY_TEAM",
    "GOALS_home",
    "SHOTS_home",
    "SHOTS_ON_TARGET_home",
    "POSSESSION_home",
    "PASS_PCT_home",
    "GOALS_away",
    "SHOTS_away",
    "SHOTS_ON_TARGET_away",
    "POSSESSION_away",
    "PASS_PCT_away",
    "HOME_WIN",
]

_EXPECTED_FEATURE_COLUMNS = [
    "GOALS_home",
    "SHOTS_home",
    "SHOTS_ON_TARGET_home",
    "POSSESSION_home",
    "PASS_PCT_home",
    "GOALS_away",
    "SHOTS_away",
    "SHOTS_ON_TARGET_away",
    "POSSESSION_away",
    "PASS_PCT_away",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_valid_matches_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "MATCH_ID": pd.array([1], dtype="int64"),
            "DATE": ["2022-12-18"],
            "HOME_TEAM": ["Brazil"],
            "AWAY_TEAM": ["France"],
            "HOME_GOALS": pd.array([2], dtype="int64"),
            "AWAY_GOALS": pd.array([1], dtype="int64"),
            "COMPETITION": ["WC"],
            "SEASON": pd.array([2022], dtype="int64"),
        }
    )


def _make_valid_train_df(n: int = 10) -> pd.DataFrame:
    """Build a valid train_data.csv DataFrame with the full schema."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        h_goals = int(rng.integers(0, 4))
        a_goals = int(rng.integers(0, 3))
        if h_goals == a_goals:
            h_goals += 1
        rows.append(
            {
                "HOME_TEAM": f"Team{i}",
                "AWAY_TEAM": f"Team{(i + 1) % n}",
                "GOALS_home": h_goals,
                "SHOTS_home": h_goals * 5 + 7,
                "SHOTS_ON_TARGET_home": max(h_goals + 2, (h_goals * 5 + 7) // 3),
                "POSSESSION_home": 55.0,
                "PASS_PCT_home": 78.0,
                "GOALS_away": a_goals,
                "SHOTS_away": a_goals * 5 + 7,
                "SHOTS_ON_TARGET_away": max(a_goals + 2, (a_goals * 5 + 7) // 3),
                "POSSESSION_away": 45.0,
                "PASS_PCT_away": 72.0,
                "HOME_WIN": int(h_goals > a_goals),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# matches.csv schema
# ---------------------------------------------------------------------------


class TestMatchesDfSchema:
    def test_valid_df_passes_validation(self) -> None:
        validate_matches_df(_make_valid_matches_df())

    def test_missing_match_id_raises(self) -> None:
        df = _make_valid_matches_df().drop(columns=["MATCH_ID"])
        with pytest.raises(ValueError, match="missing required column"):
            validate_matches_df(df)

    def test_missing_home_team_raises(self) -> None:
        df = _make_valid_matches_df().drop(columns=["HOME_TEAM"])
        with pytest.raises(ValueError, match="missing required column"):
            validate_matches_df(df)

    def test_missing_date_raises(self) -> None:
        df = _make_valid_matches_df().drop(columns=["DATE"])
        with pytest.raises(ValueError, match="missing required column"):
            validate_matches_df(df)

    def test_all_required_columns_present(self) -> None:
        df = _make_valid_matches_df()
        assert _EXPECTED_MATCHES_COLUMNS.issubset(set(df.columns))


# ---------------------------------------------------------------------------
# teams.csv schema
# ---------------------------------------------------------------------------


class TestTeamsDfSchema:
    def test_valid_df_passes_validation(self) -> None:
        df = pd.DataFrame({"TEAM_ID": pd.array([1], dtype="int64"), "NAME": ["Brazil"]})
        validate_teams_df(df)

    def test_missing_team_id_raises(self) -> None:
        df = pd.DataFrame({"NAME": ["Brazil"]})
        with pytest.raises(ValueError, match="TEAM_ID"):
            validate_teams_df(df)

    def test_team_id_as_string_raises(self) -> None:
        df = pd.DataFrame({"TEAM_ID": ["1"], "NAME": ["Brazil"]})
        with pytest.raises(ValueError, match="wrong column dtype"):
            validate_teams_df(df)


# ---------------------------------------------------------------------------
# train_data.csv column contract
# ---------------------------------------------------------------------------


class TestTrainDataSchema:
    def test_train_data_has_all_expected_columns(self) -> None:
        df = _make_valid_train_df()
        assert list(df.columns) == _EXPECTED_TRAIN_DATA_COLUMNS

    def test_train_data_has_exactly_13_columns(self) -> None:
        """HOME_TEAM + AWAY_TEAM + 10 features + HOME_WIN = 13 columns."""
        df = _make_valid_train_df()
        assert len(df.columns) == 13

    def test_train_data_feature_columns_are_numeric(self) -> None:
        df = _make_valid_train_df()
        for col in _EXPECTED_FEATURE_COLUMNS:
            assert pd.api.types.is_numeric_dtype(df[col]), (
                f"Expected {col} to be numeric, got {df[col].dtype}"
            )

    def test_home_win_is_binary(self) -> None:
        df = _make_valid_train_df()
        assert set(df["HOME_WIN"].unique()).issubset({0, 1})

    def test_home_win_matches_goals(self) -> None:
        df = _make_valid_train_df()
        expected_hw = (df["GOALS_home"] > df["GOALS_away"]).astype(int)
        pd.testing.assert_series_equal(
            df["HOME_WIN"].reset_index(drop=True),
            expected_hw.reset_index(drop=True),
            check_names=False,
        )

    def test_train_data_has_no_draws(self) -> None:
        df = _make_valid_train_df()
        assert not any(df["GOALS_home"] == df["GOALS_away"])

    def test_team_columns_are_string(self) -> None:
        df = _make_valid_train_df()
        assert pd.api.types.is_string_dtype(df["HOME_TEAM"])
        assert pd.api.types.is_string_dtype(df["AWAY_TEAM"])


# ---------------------------------------------------------------------------
# Schema validated via DataCleaner
# ---------------------------------------------------------------------------


class TestDataCleanerSchemaOutput:
    def test_clean_output_matches_train_data_schema(self) -> None:
        from worldcup_playoff.config import DataConfig, FeaturesConfig
        from worldcup_playoff.data.cleaner import DataCleaner

        matches_df = pd.DataFrame(
            {
                "MATCH_ID": pd.array([1, 2, 3], dtype="int64"),
                "DATE": ["2022-12-01", "2022-12-02", "2022-12-03"],
                "HOME_TEAM": ["Brazil", "Germany", "Spain"],
                "AWAY_TEAM": ["France", "Argentina", "England"],
                "HOME_GOALS": pd.array([2, 1, 3], dtype="int64"),
                "AWAY_GOALS": pd.array([1, 0, 1], dtype="int64"),
                "COMPETITION": ["WC", "WC", "WC"],
                "SEASON": pd.array([2022, 2022, 2022], dtype="int64"),
            }
        )
        details_df = pd.DataFrame(
            {
                "MATCH_ID": [1, 2, 3],
                "GOALS_home": [2, 1, 3],
                "SHOTS_home": [17, 12, 22],
                "SHOTS_ON_TARGET_home": [5, 4, 7],
                "POSSESSION_home": [55.0, 50.0, 60.0],
                "PASS_PCT_home": [78.0, 75.0, 80.0],
                "GOALS_away": [1, 0, 1],
                "SHOTS_away": [12, 7, 12],
                "SHOTS_ON_TARGET_away": [3, 2, 4],
                "POSSESSION_away": [45.0, 50.0, 40.0],
                "PASS_PCT_away": [72.0, 75.0, 70.0],
            }
        )

        cfg = DataConfig(
            min_date="2022-01-01",
            train_cutoff_date="2027-01-01",
        )
        cleaner = DataCleaner(cfg, FeaturesConfig())
        result = cleaner.clean(matches_df, details_df)

        assert list(result.columns) == _EXPECTED_TRAIN_DATA_COLUMNS
        assert set(result["HOME_WIN"].unique()).issubset({0, 1})
