"""Tests for DataLoader column validation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from worldcup_playoff.config import DataConfig
from worldcup_playoff.data.loader import (
    DataLoader,
    validate_matches_df,
    validate_teams_df,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_valid_matches_df() -> pd.DataFrame:
    import numpy as np

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


def _minimal_valid_teams_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TEAM_ID": pd.array([101], dtype="int64"),
            "NAME": ["Brazil"],
        }
    )


# ---------------------------------------------------------------------------
# validate_matches_df
# ---------------------------------------------------------------------------


class TestValidateMatchesDf:
    def test_valid_df_passes(self) -> None:
        validate_matches_df(_minimal_valid_matches_df())  # no exception

    def test_missing_single_column_raises(self) -> None:
        df = _minimal_valid_matches_df().drop(columns=["MATCH_ID"])
        with pytest.raises(ValueError, match="missing required column"):
            validate_matches_df(df)

    def test_missing_multiple_columns_raises(self) -> None:
        df = _minimal_valid_matches_df().drop(columns=["MATCH_ID", "DATE"])
        with pytest.raises(ValueError, match="MATCH_ID"):
            validate_matches_df(df)

    def test_close_match_suggestion_in_error(self) -> None:
        """When a column name is close to 'HOME_TEAM', the error should suggest it."""
        # Rename HOME_TEAM -> HOME_TEAMS; difflib matches at cutoff 0.7
        df = _minimal_valid_matches_df().rename(columns={"HOME_TEAM": "HOME_TEAMS"})
        with pytest.raises(ValueError, match="did you mean"):
            validate_matches_df(df)

    def test_wrong_dtype_raises(self) -> None:
        df = _minimal_valid_matches_df().copy()
        df["MATCH_ID"] = df["MATCH_ID"].astype(str)
        with pytest.raises(ValueError, match="wrong column dtype"):
            validate_matches_df(df)

    def test_numeric_dtype_compatible_does_not_raise(self) -> None:
        """int64 and float64 are compatible numeric types — should not raise."""
        df = _minimal_valid_matches_df().copy()
        df["HOME_GOALS"] = df["HOME_GOALS"].astype("float64")
        validate_matches_df(df)  # no exception


# ---------------------------------------------------------------------------
# validate_teams_df
# ---------------------------------------------------------------------------


class TestValidateTeamsDf:
    def test_valid_df_passes(self) -> None:
        validate_teams_df(_minimal_valid_teams_df())

    def test_missing_team_id_raises(self) -> None:
        df = _minimal_valid_teams_df().drop(columns=["TEAM_ID"])
        with pytest.raises(ValueError, match="TEAM_ID"):
            validate_teams_df(df)

    def test_missing_name_raises(self) -> None:
        df = _minimal_valid_teams_df().drop(columns=["NAME"])
        with pytest.raises(ValueError, match="NAME"):
            validate_teams_df(df)

    def test_team_id_as_object_raises(self) -> None:
        df = _minimal_valid_teams_df().copy()
        df["TEAM_ID"] = df["TEAM_ID"].astype(str)
        with pytest.raises(ValueError, match="wrong column dtype"):
            validate_teams_df(df)


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------


def test_load_matches_raises_when_path_is_none() -> None:
    loader = DataLoader(DataConfig(matches_path=None))
    with pytest.raises(ValueError, match="matches_path is not configured"):
        loader.load_matches()


def test_load_teams_raises_when_path_is_none() -> None:
    loader = DataLoader(DataConfig(teams_path=None))
    with pytest.raises(ValueError, match="teams_path is not configured"):
        loader.load_teams()


@patch("worldcup_playoff.data.loader.validate_matches_df")
def test_load_matches_reads_csv(mock_validate: object, tmp_path: Path) -> None:
    csv_dir = tmp_path / "dataset" / "csv"
    csv_dir.mkdir(parents=True)
    csv_file = csv_dir / "matches.csv"
    csv_file.write_text("MATCH_ID,HOME_TEAM\n1,Brazil\n")

    cfg = DataConfig(matches_path="dataset/csv/matches.csv")
    loader = DataLoader(cfg, root=tmp_path)
    df = loader.load_matches()
    assert len(df) == 1
    assert "MATCH_ID" in df.columns


@patch("worldcup_playoff.data.loader.validate_teams_df")
def test_load_teams_reads_csv(mock_validate: object, tmp_path: Path) -> None:
    csv_dir = tmp_path / "dataset" / "csv"
    csv_dir.mkdir(parents=True)
    csv_file = csv_dir / "teams.csv"
    csv_file.write_text("TEAM_ID,NAME\n101,Brazil\n")

    cfg = DataConfig(teams_path="dataset/csv/teams.csv")
    loader = DataLoader(cfg, root=tmp_path)
    df = loader.load_teams()
    assert len(df) == 1
    assert "NAME" in df.columns


def test_load_matches_raises_on_schema_mismatch(tmp_path: Path) -> None:
    csv_dir = tmp_path / "dataset" / "csv"
    csv_dir.mkdir(parents=True)
    (csv_dir / "matches.csv").write_text("FOO,BAR\n1,x\n")

    cfg = DataConfig(matches_path="dataset/csv/matches.csv")
    loader = DataLoader(cfg, root=tmp_path)
    with pytest.raises(ValueError, match="missing required column"):
        loader.load_matches()


def test_load_teams_raises_on_schema_mismatch(tmp_path: Path) -> None:
    csv_dir = tmp_path / "dataset" / "csv"
    csv_dir.mkdir(parents=True)
    (csv_dir / "teams.csv").write_text("FOO,BAR\n1,x\n")

    cfg = DataConfig(teams_path="dataset/csv/teams.csv")
    loader = DataLoader(cfg, root=tmp_path)
    with pytest.raises(ValueError, match="missing required column"):
        loader.load_teams()
