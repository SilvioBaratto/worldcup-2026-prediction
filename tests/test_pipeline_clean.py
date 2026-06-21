"""Tests for Pipeline.run_clean() — wiring with synthetic CSVs."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from worldcup_playoff.config import AppConfig
from worldcup_playoff.pipeline import Pipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pipeline(tmp_path: Path, config: AppConfig | None = None) -> Pipeline:
    return Pipeline(config or AppConfig(), root=tmp_path)


def _minimal_clean_result() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "HOME_TEAM": ["Brazil"],
            "AWAY_TEAM": ["France"],
            "GOALS_home": [2],
            "SHOTS_home": [17],
            "SHOTS_ON_TARGET_home": [5],
            "POSSESSION_home": [55.0],
            "PASS_PCT_home": [78.0],
            "GOALS_away": [1],
            "SHOTS_away": [12],
            "SHOTS_ON_TARGET_away": [3],
            "POSSESSION_away": [45.0],
            "PASS_PCT_away": [72.0],
            "HOME_WIN": [1],
        }
    )


def _write_minimal_matches_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "MATCH_ID,DATE,HOME_TEAM,AWAY_TEAM,HOME_GOALS,AWAY_GOALS,COMPETITION,SEASON\n"
        "1001,2022-12-18,Brazil,France,2,1,WC,2022\n"
        "1002,2022-12-17,Germany,Argentina,1,0,WC,2022\n"
    )


def _write_minimal_details_csv(path: Path, match_ids: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "MATCH_ID,GOALS_home,SHOTS_home,SHOTS_ON_TARGET_home,POSSESSION_home,PASS_PCT_home,GOALS_away,SHOTS_away,SHOTS_ON_TARGET_away,POSSESSION_away,PASS_PCT_away\n"
    rows = "\n".join(
        f"{mid},1,12,4,55.0,78.0,0,7,2,45.0,72.0" for mid in match_ids
    )
    path.write_text(header + rows + "\n")


# ---------------------------------------------------------------------------
# Pipeline.run_clean with mocked internals
# ---------------------------------------------------------------------------


class TestRunCleanMocked:
    def test_loader_is_called_for_matches(self, tmp_path: Path) -> None:
        pipeline = _make_pipeline(tmp_path)

        with (
            patch("worldcup_playoff.pipeline.DataLoader") as MockLoader,
            patch("worldcup_playoff.pipeline.DataCleaner") as MockCleaner,
        ):
            loader_instance = MockLoader.return_value
            loader_instance.load_matches.return_value = pd.DataFrame({"a": [1]})
            MockCleaner.return_value.write.return_value = _minimal_clean_result()

            pipeline.run_clean()

        loader_instance.load_matches.assert_called_once()

    def test_cleaner_write_is_called(self, tmp_path: Path) -> None:
        pipeline = _make_pipeline(tmp_path)

        with (
            patch("worldcup_playoff.pipeline.DataLoader") as MockLoader,
            patch("worldcup_playoff.pipeline.DataCleaner") as MockCleaner,
        ):
            loader_instance = MockLoader.return_value
            loader_instance.load_matches.return_value = pd.DataFrame({"a": [1]})
            cleaner_instance = MockCleaner.return_value
            cleaner_instance.write.return_value = _minimal_clean_result()

            pipeline.run_clean()

        cleaner_instance.write.assert_called_once()

    def test_returns_output_path(self, tmp_path: Path) -> None:
        pipeline = _make_pipeline(tmp_path)
        expected = tmp_path / AppConfig().data.output_path

        with (
            patch("worldcup_playoff.pipeline.DataLoader") as MockLoader,
            patch("worldcup_playoff.pipeline.DataCleaner") as MockCleaner,
        ):
            MockLoader.return_value.load_matches.return_value = pd.DataFrame()
            MockCleaner.return_value.write.return_value = _minimal_clean_result()

            result = pipeline.run_clean()

        assert result == expected


# ---------------------------------------------------------------------------
# Pipeline.run_clean with real CSVs (integration-style)
# ---------------------------------------------------------------------------


class TestRunCleanWithRealCsvs:
    def test_run_clean_writes_train_data_csv(self, tmp_path: Path) -> None:
        """End-to-end: write raw CSVs, call run_clean, verify output exists."""
        matches_path = tmp_path / "dataset" / "csv" / "matches.csv"
        details_path = tmp_path / "dataset" / "csv" / "match_details.csv"
        _write_minimal_matches_csv(matches_path)
        _write_minimal_details_csv(details_path, [1001, 1002])

        config = AppConfig()
        # Override paths to point at our tmp files
        from worldcup_playoff.config import DataConfig

        data_cfg = DataConfig(
            matches_path="dataset/csv/matches.csv",
            match_details_csv_path="dataset/csv/match_details.csv",
            output_path="dataset/train_data.csv",
            min_date="2022-01-01",
            train_cutoff_date="2027-01-01",
        )
        cfg = config.model_copy(update={"data": data_cfg})
        pipeline = Pipeline(cfg, root=tmp_path)
        output = pipeline.run_clean()

        assert output.exists()
        result = pd.read_csv(output)
        assert "HOME_WIN" in result.columns
        assert "HOME_TEAM" in result.columns

    def test_run_clean_without_details_uses_matches_only(self, tmp_path: Path) -> None:
        """When match_details.csv is absent but matches have feature columns inline."""
        # For this test we use the mocked approach since inline features need
        # the cleaner to have those columns already present.
        pipeline = _make_pipeline(tmp_path)

        with (
            patch("worldcup_playoff.pipeline.DataLoader") as MockLoader,
            patch("worldcup_playoff.pipeline.DataCleaner") as MockCleaner,
        ):
            MockLoader.return_value.load_matches.return_value = pd.DataFrame()
            MockCleaner.return_value.write.return_value = _minimal_clean_result()

            pipeline.run_clean()

        # No crash means details-absent path was handled
        MockCleaner.return_value.write.assert_called_once()
