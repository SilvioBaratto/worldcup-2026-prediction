"""Source-blind spec tests for issue #27: no-key martj42 historical loader.

Tests are derived exclusively from the acceptance criteria and requirements.md.
This file supplements tests/test_martj42_loader.py with coverage of three gaps
that are absent from the existing file:

  Gap 1 — MINUTE→Int64 coercion in goalscorers (criterion 3 is explicit but untested).
  Gap 2 — ValueError message content: "clear … with close-match hints" (criterion 6 only
           has pytest.raises(ValueError) with no message assertion in the existing file).
  Gap 3 — Cache-miss download + write-to-disk path (criterion 1: "else downloads from
           base_url and writes the cache"). The existing file tests only the cache-HIT
           half; the download half is completely absent.

Skipped criteria (per oracle):
  - "Team-name columns are normalized through the crosswalk" — NOT VERIFIABLE per oracle.
  - "All tests pass" — boilerplate suite gate; no per-criterion assertion.
  - "SOLID, clean code" — subjective prose; no concrete runtime assertion.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from worldcup_playoff.config import Martj42Config
from worldcup_playoff.data.martj42_loader import (
    Martj42Loader,
    validate_goalscorers_df,
    validate_results_df,
    validate_shootouts_df,
)


# ===========================================================================
# Minimal fixture CSVs — built from data-contract shape in requirements.md,
# never from implementation source.
# ===========================================================================

_RESULTS_CSV = """\
date,home_team,away_team,home_score,away_score,tournament,city,country,neutral
2006-06-09,Germany,Costa Rica,4,2,FIFA World Cup,Munich,Germany,FALSE
2018-07-15,France,Croatia,4,2,FIFA World Cup,Moscow,Russia,TRUE
2026-06-11,United States,Mexico,NA,NA,FIFA World Cup,New York,United States,TRUE
2026-06-12,Brazil,Saudi Arabia,3,0,FIFA World Cup,Los Angeles,United States,TRUE
2025-03-26,England,Albania,2,0,UEFA Nations League,London,England,FALSE
"""

_SHOOTOUTS_CSV = """\
date,home_team,away_team,winner,first_shooter
2018-07-07,Russia,Croatia,Croatia,Russia
"""

_GOALSCORERS_CSV = """\
date,home_team,away_team,team,scorer,minute,own_goal,penalty
2006-06-09,Germany,Costa Rica,Germany,Lahm,6,FALSE,FALSE
2006-06-09,Germany,Costa Rica,Costa Rica,Wanchope,12,FALSE,FALSE
2018-06-16,IR Iran,Morocco,IR Iran,Bouhaddouz,90,TRUE,FALSE
"""


@pytest.fixture()
def populated_cache(tmp_path: Path) -> Path:
    """Write all three martj42 CSVs into tmp_path so no HTTP is needed."""
    (tmp_path / "results.csv").write_text(_RESULTS_CSV)
    (tmp_path / "shootouts.csv").write_text(_SHOOTOUTS_CSV)
    (tmp_path / "goalscorers.csv").write_text(_GOALSCORERS_CSV)
    return tmp_path


@pytest.fixture()
def loader(populated_cache: Path) -> Martj42Loader:
    return Martj42Loader(config=Martj42Config(cache_dir=populated_cache))


# ===========================================================================
# DataFrame builders for validator tests — shapes taken from criteria only.
# ===========================================================================


def _valid_results_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "DATE": ["2006-06-09", "2026-06-11"],
            "HOME_TEAM": ["Germany", "United States"],
            "AWAY_TEAM": ["Costa Rica", "Mexico"],
            "HOME_GOALS": pd.array([4, pd.NA], dtype="Int64"),
            "AWAY_GOALS": pd.array([2, pd.NA], dtype="Int64"),
            "TOURNAMENT": ["FIFA World Cup", "FIFA World Cup"],
            "NEUTRAL": [False, True],
        }
    )


def _valid_shootouts_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "DATE": ["2018-07-07"],
            "HOME_TEAM": ["Russia"],
            "AWAY_TEAM": ["Croatia"],
            "WINNER": ["Croatia"],
            "FIRST_SHOOTER": ["Russia"],
        }
    )


def _valid_goalscorers_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "DATE": ["2006-06-09"],
            "HOME_TEAM": ["Germany"],
            "AWAY_TEAM": ["Costa Rica"],
            "TEAM": ["Germany"],
            "SCORER": ["Lahm"],
            "MINUTE": pd.array([6], dtype="Int64"),
            "OWN_GOAL": [False],
            "PENALTY": [False],
        }
    )


# ===========================================================================
# Gap 1 — Criterion 3: MINUTE→Int64 coercion in goalscorers
# (the existing test file asserts OWN_GOAL and PENALTY dtypes but NOT MINUTE)
# ===========================================================================


def test_when_goalscorers_loaded_then_minute_dtype_is_nullable_int64(loader: Martj42Loader):
    """Criterion: 'goalscorers coerces MINUTE→Int64'."""
    df = loader.load_goalscorers()
    assert df["MINUTE"].dtype == pd.Int64Dtype(), (
        f"MINUTE must be Int64 (nullable), got {df['MINUTE'].dtype}"
    )


def test_when_goalscorers_loaded_then_minute_values_are_integers(loader: Martj42Loader):
    """MINUTE values from the fixture CSV (6, 12, 90) are integer-valued in the output."""
    df = loader.load_goalscorers()
    non_na = df["MINUTE"].dropna()
    assert len(non_na) == 3
    # pd.api.types.is_integer checks scalar value type within a nullable Int64 column
    assert all(pd.api.types.is_integer(v) for v in non_na)


def test_when_goalscorers_minute_is_large_then_it_stays_within_int64_range(loader: Martj42Loader):
    """Any valid minute (≤ 130 for AET) must survive the Int64 coercion without overflow."""
    df = loader.load_goalscorers()
    non_na = df["MINUTE"].dropna()
    assert all(v >= 0 for v in non_na)


# ===========================================================================
# Gap 2 — Criterion 6: ValueError message content with close-match hints
# The existing file only asserts pytest.raises(ValueError); it never checks
# that the message is "clear" or contains "close-match hints".
# ===========================================================================


def test_when_results_df_missing_home_goals_then_error_message_names_home_goals():
    """A 'clear ValueError' must name the missing column so users know what to fix."""
    df = _valid_results_df().drop(columns=["HOME_GOALS"])
    with pytest.raises(ValueError, match=r"HOME_GOALS"):
        validate_results_df(df)


def test_when_results_df_missing_neutral_then_error_message_names_neutral():
    df = _valid_results_df().drop(columns=["NEUTRAL"])
    with pytest.raises(ValueError, match=r"NEUTRAL"):
        validate_results_df(df)


def test_when_results_df_missing_tournament_then_error_message_names_tournament():
    df = _valid_results_df().drop(columns=["TOURNAMENT"])
    with pytest.raises(ValueError, match=r"TOURNAMENT"):
        validate_results_df(df)


def test_when_shootouts_df_missing_winner_then_error_message_names_winner():
    df = _valid_shootouts_df().drop(columns=["WINNER"])
    with pytest.raises(ValueError, match=r"WINNER"):
        validate_shootouts_df(df)


def test_when_goalscorers_df_missing_minute_then_error_message_names_minute():
    df = _valid_goalscorers_df().drop(columns=["MINUTE"])
    with pytest.raises(ValueError, match=r"MINUTE"):
        validate_goalscorers_df(df)


def test_when_goalscorers_df_missing_penalty_then_error_message_names_penalty():
    df = _valid_goalscorers_df().drop(columns=["PENALTY"])
    with pytest.raises(ValueError, match=r"PENALTY"):
        validate_goalscorers_df(df)


def test_when_results_df_has_wrong_dtype_for_home_goals_then_error_message_is_non_empty():
    """Dtype-mismatch must produce a non-empty human-readable message, not a bare ValueError."""
    df = _valid_results_df().copy()
    df["HOME_GOALS"] = df["HOME_GOALS"].astype(float)
    with pytest.raises(ValueError) as exc_info:
        validate_results_df(df)
    assert len(str(exc_info.value).strip()) > 0


def test_when_results_df_has_wrong_dtype_for_neutral_then_error_message_is_non_empty():
    df = _valid_results_df().copy()
    df["NEUTRAL"] = df["NEUTRAL"].astype(str)
    with pytest.raises(ValueError) as exc_info:
        validate_results_df(df)
    assert len(str(exc_info.value).strip()) > 0


# ===========================================================================
# Gap 3 — Criterion 1: cache-miss downloads from base_url and writes the cache
# The existing file only tests the cache-HIT half (no HTTP when cache exists).
# These tests cover the cache-MISS half: HTTP is called, file is written to disk.
#
# Assumption: the loader uses `requests.get` (standard Python HTTP library).
# If the implementation uses urllib or httpx, the patch target will differ and
# these tests will red-flag that discrepancy.
# ===========================================================================


def _fake_response(csv_text: str) -> MagicMock:
    resp = MagicMock()
    resp.text = csv_text
    resp.content = csv_text.encode()
    resp.raise_for_status.return_value = None
    return resp


def test_when_results_csv_absent_from_cache_then_http_get_is_called(tmp_path: Path):
    """Cache-miss: when results.csv is not in cache_dir, an HTTP download is triggered."""
    ldr = Martj42Loader(config=Martj42Config(cache_dir=tmp_path))
    with patch("requests.get", return_value=_fake_response(_RESULTS_CSV)) as mock_get:
        ldr.load_results()
    assert mock_get.called, "requests.get must be called when results.csv is absent from cache"


def test_when_shootouts_csv_absent_from_cache_then_http_get_is_called(tmp_path: Path):
    ldr = Martj42Loader(config=Martj42Config(cache_dir=tmp_path))
    with patch("requests.get", return_value=_fake_response(_SHOOTOUTS_CSV)) as mock_get:
        ldr.load_shootouts()
    assert mock_get.called, "requests.get must be called when shootouts.csv is absent from cache"


def test_when_goalscorers_csv_absent_from_cache_then_http_get_is_called(tmp_path: Path):
    ldr = Martj42Loader(config=Martj42Config(cache_dir=tmp_path))
    with patch("requests.get", return_value=_fake_response(_GOALSCORERS_CSV)) as mock_get:
        ldr.load_goalscorers()
    assert mock_get.called, "requests.get must be called when goalscorers.csv is absent from cache"


def test_when_results_downloaded_then_cache_file_is_written_to_cache_dir(tmp_path: Path):
    """Cache-write: after a download, results.csv must exist in cache_dir."""
    ldr = Martj42Loader(config=Martj42Config(cache_dir=tmp_path))
    with patch("requests.get", return_value=_fake_response(_RESULTS_CSV)):
        ldr.load_results()
    assert (tmp_path / "results.csv").exists(), (
        "results.csv must be written to cache_dir after a successful download"
    )


def test_when_shootouts_downloaded_then_cache_file_is_written_to_cache_dir(tmp_path: Path):
    ldr = Martj42Loader(config=Martj42Config(cache_dir=tmp_path))
    with patch("requests.get", return_value=_fake_response(_SHOOTOUTS_CSV)):
        ldr.load_shootouts()
    assert (tmp_path / "shootouts.csv").exists()


def test_when_goalscorers_downloaded_then_cache_file_is_written_to_cache_dir(tmp_path: Path):
    ldr = Martj42Loader(config=Martj42Config(cache_dir=tmp_path))
    with patch("requests.get", return_value=_fake_response(_GOALSCORERS_CSV)):
        ldr.load_goalscorers()
    assert (tmp_path / "goalscorers.csv").exists()


def test_when_results_downloaded_and_cached_then_second_call_skips_http(tmp_path: Path):
    """Cache-miss then cache-hit: after the first download writes the cache, the second
    call must not issue any further HTTP requests — it must read from disk."""
    ldr = Martj42Loader(config=Martj42Config(cache_dir=tmp_path))

    # First call: should download and write cache
    with patch("requests.get", return_value=_fake_response(_RESULTS_CSV)):
        ldr.load_results()

    # Second call: cache exists; HTTP must NOT be called
    with patch(
        "requests.get",
        side_effect=AssertionError("HTTP must not be issued after cache is populated"),
    ) as mock_get:
        ldr.load_results()
    mock_get.assert_not_called()


def test_when_results_cache_absent_then_downloaded_data_matches_returned_dataframe(tmp_path: Path):
    """The DataFrame returned on a cache-miss must reflect what was downloaded, not garbage."""
    ldr = Martj42Loader(config=Martj42Config(cache_dir=tmp_path))
    with patch("requests.get", return_value=_fake_response(_RESULTS_CSV)):
        df = ldr.load_results()
    # The fixture CSV has 5 data rows; Germany scored 4 home goals in row 1.
    assert len(df) == 5
    germany_row = df[df["HOME_TEAM"] == "Germany"]
    assert not germany_row.empty
    assert germany_row["HOME_GOALS"].iloc[0] == 4


# ===========================================================================
# Property-based tests
# ===========================================================================


# --- Property: missing-column error names the absent column (invariant across all columns) ---

_RESULTS_COLS = [
    "DATE",
    "HOME_TEAM",
    "AWAY_TEAM",
    "HOME_GOALS",
    "AWAY_GOALS",
    "TOURNAMENT",
    "NEUTRAL",
]

_SHOOTOUTS_COLS = ["DATE", "HOME_TEAM", "AWAY_TEAM", "WINNER", "FIRST_SHOOTER"]

_GOALSCORERS_COLS = [
    "DATE",
    "HOME_TEAM",
    "AWAY_TEAM",
    "TEAM",
    "SCORER",
    "MINUTE",
    "OWN_GOAL",
    "PENALTY",
]


@given(st.sampled_from(_RESULTS_COLS))
def test_when_any_results_column_is_absent_then_error_message_names_that_column(missing: str):
    """Close-match-hint invariant: for every required results column, its name must appear
    in the ValueError message when that column is dropped."""
    df = _valid_results_df().drop(columns=[missing])
    with pytest.raises(ValueError, match=missing):
        validate_results_df(df)


@given(st.sampled_from(_SHOOTOUTS_COLS))
def test_when_any_shootouts_column_is_absent_then_error_message_names_that_column(missing: str):
    df = _valid_shootouts_df().drop(columns=[missing])
    with pytest.raises(ValueError, match=missing):
        validate_shootouts_df(df)


@given(st.sampled_from(_GOALSCORERS_COLS))
def test_when_any_goalscorers_column_is_absent_then_error_message_names_that_column(missing: str):
    df = _valid_goalscorers_df().drop(columns=[missing])
    with pytest.raises(ValueError, match=missing):
        validate_goalscorers_df(df)


# --- Property: load_results() is idempotent given a stable cache ---


@settings(max_examples=1)
@given(st.just(None))
def test_when_load_results_called_twice_on_stable_cache_then_results_are_identical(_: None):
    """Idempotence: two successive load_results() calls return DataFrames with identical content."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        cache = Path(d)
        (cache / "results.csv").write_text(_RESULTS_CSV)
        (cache / "shootouts.csv").write_text(_SHOOTOUTS_CSV)
        (cache / "goalscorers.csv").write_text(_GOALSCORERS_CSV)
        ldr = Martj42Loader(config=Martj42Config(cache_dir=cache))
        df1 = ldr.load_results()
        df2 = ldr.load_results()
    pd.testing.assert_frame_equal(df1, df2)
