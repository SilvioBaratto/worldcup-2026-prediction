"""Tests for DataCleaner."""

from __future__ import annotations

import pandas as pd
import pytest

from worldcup_playoff.config import DataConfig, FeaturesConfig
from worldcup_playoff.data.cleaner import DataCleaner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cleaner(
    min_date: str = "2006-01-01",
    train_cutoff_date: str = "2027-01-01",
    epsilon: float = 0.001,
) -> DataCleaner:
    cfg = DataConfig(
        min_date=min_date,
        train_cutoff_date=train_cutoff_date,
        epsilon=epsilon,
    )
    return DataCleaner(cfg, FeaturesConfig())


def _minimal_matches_df(n: int = 5) -> pd.DataFrame:
    """Minimal matches.csv DataFrame with no feature columns (to be merged with details)."""
    import numpy as np

    rows = []
    for i in range(n):
        home_g = i % 3 + 1
        away_g = i % 2
        if home_g == away_g:
            away_g = 0
        rows.append(
            {
                "MATCH_ID": 1000 + i,
                "DATE": f"2022-0{i + 1}-01",
                "HOME_TEAM": f"TeamA{i}",
                "AWAY_TEAM": f"TeamB{i}",
                "HOME_GOALS": home_g,
                "AWAY_GOALS": away_g,
                "COMPETITION": "WC",
                "SEASON": 2022,
            }
        )
    return pd.DataFrame(rows).astype(
        {
            "MATCH_ID": "int64",
            "HOME_GOALS": "int64",
            "AWAY_GOALS": "int64",
            "SEASON": "int64",
        }
    )


def _minimal_details_df(match_ids: list[int]) -> pd.DataFrame:
    """Minimal match_details.csv for the given match IDs."""
    rows = []
    for mid in match_ids:
        rows.append(
            {
                "MATCH_ID": mid,
                "GOALS_home": 1,
                "SHOTS_home": 12,
                "SHOTS_ON_TARGET_home": 4,
                "POSSESSION_home": 55.0,
                "PASS_PCT_home": 78.0,
                "GOALS_away": 0,
                "SHOTS_away": 7,
                "SHOTS_ON_TARGET_away": 2,
                "POSSESSION_away": 45.0,
                "PASS_PCT_away": 72.0,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Individual pipeline steps
# ---------------------------------------------------------------------------


class TestDataCleanerSteps:
    def test_sort_by_date(self) -> None:
        cleaner = _make_cleaner()
        df = _minimal_matches_df(3)
        df = df.sort_values("DATE", ascending=False)  # reverse order
        result = cleaner._sort_by_date(df)
        assert result["DATE"].tolist() == sorted(result["DATE"].tolist())

    def test_filter_by_min_date_keeps_rows_after_min(self) -> None:
        cleaner = _make_cleaner(min_date="2022-03-01")
        df = _minimal_matches_df(5)  # dates: 2022-01 to 2022-05
        result = cleaner._filter_by_min_date(df)
        assert all(d >= "2022-03-01" for d in result["DATE"])

    def test_filter_by_min_date_drops_older_rows(self) -> None:
        cleaner = _make_cleaner(min_date="2023-01-01")
        df = _minimal_matches_df(5)
        result = cleaner._filter_by_min_date(df)
        assert len(result) == 0

    def test_drop_draws_removes_equal_goals(self) -> None:
        cleaner = _make_cleaner()
        df = _minimal_matches_df(3)
        df.loc[0, "HOME_GOALS"] = 1
        df.loc[0, "AWAY_GOALS"] = 1  # draw
        result = cleaner._drop_draws(df)
        assert len(result) == len(df) - 1
        assert not any(result["HOME_GOALS"] == result["AWAY_GOALS"])

    def test_drop_draws_keeps_non_drawn_matches(self) -> None:
        cleaner = _make_cleaner()
        df = _minimal_matches_df(4)
        # Ensure no draws in sample
        df["HOME_GOALS"] = [1, 2, 1, 3]
        df["AWAY_GOALS"] = [0, 1, 0, 0]
        result = cleaner._drop_draws(df)
        assert len(result) == 4

    def test_add_home_win_when_home_goals_greater(self) -> None:
        cleaner = _make_cleaner()
        df = _minimal_matches_df(1)
        df["HOME_GOALS"] = [2]
        df["AWAY_GOALS"] = [1]
        df = df.assign(
            GOALS_home=[2],
            SHOTS_home=[17],
            SHOTS_ON_TARGET_home=[5],
            POSSESSION_home=[55.0],
            PASS_PCT_home=[78.0],
            GOALS_away=[1],
            SHOTS_away=[12],
            SHOTS_ON_TARGET_away=[3],
            POSSESSION_away=[45.0],
            PASS_PCT_away=[72.0],
        )
        result = cleaner._add_home_win(df)
        assert result["HOME_WIN"].iloc[0] == 1

    def test_add_home_win_when_away_goals_greater(self) -> None:
        cleaner = _make_cleaner()
        df = _minimal_matches_df(1)
        df["HOME_GOALS"] = [0]
        df["AWAY_GOALS"] = [2]
        df = df.assign(
            GOALS_home=[0],
            SHOTS_home=[7],
            SHOTS_ON_TARGET_home=[2],
            POSSESSION_home=[45.0],
            PASS_PCT_home=[72.0],
            GOALS_away=[2],
            SHOTS_away=[17],
            SHOTS_ON_TARGET_away=[5],
            POSSESSION_away=[55.0],
            PASS_PCT_away=[78.0],
        )
        result = cleaner._add_home_win(df)
        assert result["HOME_WIN"].iloc[0] == 0

    def test_fix_zero_percentages_replaces_zero_with_epsilon(self) -> None:
        cleaner = _make_cleaner(epsilon=0.001)
        df = _minimal_matches_df(1)
        df = df.assign(
            GOALS_home=[1], SHOTS_home=[12], SHOTS_ON_TARGET_home=[4],
            POSSESSION_home=[0.0], PASS_PCT_home=[0.0],  # zeros to be fixed
            GOALS_away=[0], SHOTS_away=[7], SHOTS_ON_TARGET_away=[2],
            POSSESSION_away=[0.0], PASS_PCT_away=[0.0],
        )
        result = cleaner._fix_zero_percentages(df)
        assert result.iloc[0]["POSSESSION_home"] == pytest.approx(0.001)
        assert result.iloc[0]["PASS_PCT_home"] == pytest.approx(0.001)
        assert result.iloc[0]["POSSESSION_away"] == pytest.approx(0.001)
        assert result.iloc[0]["PASS_PCT_away"] == pytest.approx(0.001)

    def test_apply_train_cutoff_keeps_only_rows_before_cutoff(self) -> None:
        cleaner = _make_cleaner(train_cutoff_date="2022-04-01")
        df = _minimal_matches_df(5)  # dates 2022-01 to 2022-05
        result = cleaner._apply_train_cutoff(df)
        assert all(d < "2022-04-01" for d in result["DATE"])


# ---------------------------------------------------------------------------
# Full DataCleaner.clean pipeline
# ---------------------------------------------------------------------------


class TestDataCleanerCleanPipeline:
    def _make_full_df(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return (matches_df, details_df) for full pipeline testing."""
        matches_df = _minimal_matches_df(5)
        # Make sure we have no draws
        matches_df["HOME_GOALS"] = [2, 1, 3, 2, 1]
        matches_df["AWAY_GOALS"] = [0, 0, 1, 0, 0]
        details_df = _minimal_details_df(matches_df["MATCH_ID"].tolist())
        return matches_df, details_df

    def test_clean_output_has_correct_columns(self) -> None:
        cleaner = _make_cleaner()
        matches_df, details_df = self._make_full_df()
        result = cleaner.clean(matches_df, details_df)
        expected_cols = [
            "HOME_TEAM", "AWAY_TEAM",
            "GOALS_home", "SHOTS_home", "SHOTS_ON_TARGET_home", "POSSESSION_home", "PASS_PCT_home",
            "GOALS_away", "SHOTS_away", "SHOTS_ON_TARGET_away", "POSSESSION_away", "PASS_PCT_away",
            "HOME_WIN",
        ]
        assert list(result.columns) == expected_cols

    def test_clean_home_win_is_binary(self) -> None:
        cleaner = _make_cleaner()
        matches_df, details_df = self._make_full_df()
        result = cleaner.clean(matches_df, details_df)
        assert set(result["HOME_WIN"].unique()).issubset({0, 1})

    def test_clean_drops_draws(self) -> None:
        cleaner = _make_cleaner()
        matches_df = _minimal_matches_df(4)
        matches_df["HOME_GOALS"] = [2, 1, 1, 3]
        matches_df["AWAY_GOALS"] = [1, 1, 0, 1]  # row 1 is a draw
        details_df = _minimal_details_df(matches_df["MATCH_ID"].tolist())
        result = cleaner.clean(matches_df, details_df)
        assert not any(result["HOME_WIN"].isna())
        # Only 3 non-draw matches should remain
        assert len(result) == 3

    def test_clean_merges_details(self) -> None:
        cleaner = _make_cleaner()
        matches_df, details_df = self._make_full_df()
        result = cleaner.clean(matches_df, details_df)
        assert "SHOTS_home" in result.columns

    def test_clean_raises_without_details_and_missing_features(self) -> None:
        cleaner = _make_cleaner()
        matches_df = _minimal_matches_df(3)
        # No feature columns and no details_df — must raise
        with pytest.raises(ValueError, match="Feature columns"):
            cleaner.clean(matches_df, details_df=None)

    def test_write_saves_to_csv(self, tmp_path: Path) -> None:
        import pandas as pd

        cleaner = _make_cleaner()
        matches_df, details_df = self._make_full_df()

        # Configure output path inside tmp_path
        cfg = DataConfig(
            min_date="2006-01-01",
            train_cutoff_date="2027-01-01",
            output_path="dataset/train_data.csv",
        )
        dc = DataCleaner(cfg, FeaturesConfig())
        dc.write(matches_df, details_df=details_df, root=tmp_path)

        output = tmp_path / "dataset" / "train_data.csv"
        assert output.exists()
        loaded = pd.read_csv(output)
        assert "HOME_WIN" in loaded.columns
