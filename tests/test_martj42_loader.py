"""Source-blind example tests for issue #2: no-key martj42 historical loader.

Tests are derived exclusively from the acceptance criteria and requirements.md.
No implementation source was read during authoring.

Skipped criteria (per oracle):
  - "All tests pass" — boilerplate suite gate, no per-criterion assertion.
  - "SOLID, clean code" — subjective prose, no concrete runtime assertion.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from hypothesis import given, strategies as st

# --- Modules under test (do not exist yet — tests are expected to fail until implemented) ---
from worldcup_playoff.data.martj42_loader import (
    REQUIRED_MARTJ42_GOALSCORERS_COLUMNS,
    REQUIRED_MARTJ42_RESULTS_COLUMNS,
    REQUIRED_MARTJ42_SHOOTOUTS_COLUMNS,
    Martj42Loader,
    validate_goalscorers_df,
    validate_results_df,
    validate_shootouts_df,
    wc2026_schedule,
)
from worldcup_playoff.config import AppConfig, Martj42Config


# =============================================================================
# Fixture helpers — raw martj42 CSV text, never touching production code
# =============================================================================

_RESULTS_CSV = """\
date,home_team,away_team,home_score,away_score,tournament,city,country,neutral
2006-06-09,Germany,Costa Rica,4,2,FIFA World Cup,Munich,Germany,FALSE
2018-07-15,France,Croatia,4,2,FIFA World Cup,Moscow,Russia,TRUE
2018-06-16,IR Iran,Morocco,1,0,FIFA World Cup,Saint Petersburg,Russia,TRUE
2026-06-11,United States,Mexico,NA,NA,FIFA World Cup,New York,United States,TRUE
2026-06-12,Brazil,Saudi Arabia,3,0,FIFA World Cup,Los Angeles,United States,TRUE
2025-03-26,England,Albania,2,0,UEFA Nations League,London,England,FALSE
"""

_SHOOTOUTS_CSV = """\
date,home_team,away_team,winner,first_shooter
2018-07-07,Russia,Croatia,Croatia,Russia
2018-07-06,Sweden,England,England,Sweden
"""

_GOALSCORERS_CSV = """\
date,home_team,away_team,team,scorer,minute,own_goal,penalty
2006-06-09,Germany,Costa Rica,Germany,Lahm,6,FALSE,FALSE
2006-06-09,Germany,Costa Rica,Costa Rica,Wanchope,12,FALSE,FALSE
2018-06-16,IR Iran,Morocco,IR Iran,Bouhaddouz,90,TRUE,FALSE
"""


@pytest.fixture()
def cache_dir(tmp_path: Path) -> Path:
    """Populate tmp_path with martj42 fixture CSVs; loader reads from here, no HTTP needed."""
    (tmp_path / "results.csv").write_text(_RESULTS_CSV)
    (tmp_path / "shootouts.csv").write_text(_SHOOTOUTS_CSV)
    (tmp_path / "goalscorers.csv").write_text(_GOALSCORERS_CSV)
    return tmp_path


@pytest.fixture()
def loader(cache_dir: Path) -> Martj42Loader:
    return Martj42Loader(config=Martj42Config(cache_dir=cache_dir))


# =============================================================================
# Builder helpers for validator tests — constructed from criteria, not from src
# =============================================================================


def _make_valid_results_df() -> pd.DataFrame:
    """Minimal DataFrame matching the internal results schema from the criteria."""
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


def _make_valid_shootouts_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "DATE": ["2018-07-07"],
            "HOME_TEAM": ["Russia"],
            "AWAY_TEAM": ["Croatia"],
            "WINNER": ["Croatia"],
            "FIRST_SHOOTER": ["Russia"],
        }
    )


def _make_valid_goalscorers_df() -> pd.DataFrame:
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


# =============================================================================
# Criterion 1 — load_results() schema and dtype coercions
# =============================================================================


def test_when_results_loaded_then_exact_seven_columns_are_returned(loader: Martj42Loader):
    df = loader.load_results()
    expected = {
        "DATE",
        "HOME_TEAM",
        "AWAY_TEAM",
        "HOME_GOALS",
        "AWAY_GOALS",
        "TOURNAMENT",
        "NEUTRAL",
    }
    assert set(df.columns) == expected


def test_when_results_loaded_then_no_extra_columns_beyond_contract_are_present(
    loader: Martj42Loader,
):
    # city and country (present in martj42 source) must be dropped
    df = loader.load_results()
    assert len(df.columns) == 7


def test_when_results_loaded_then_home_goals_dtype_is_int64_nullable(loader: Martj42Loader):
    df = loader.load_results()
    assert df["HOME_GOALS"].dtype == pd.Int64Dtype()


def test_when_results_loaded_then_away_goals_dtype_is_int64_nullable(loader: Martj42Loader):
    df = loader.load_results()
    assert df["AWAY_GOALS"].dtype == pd.Int64Dtype()


def test_when_score_is_na_string_in_csv_then_home_goals_becomes_pandas_na(loader: Martj42Loader):
    # The 2026 fixture has home_score == "NA" in the raw martj42 file
    df = loader.load_results()
    unplayed = df[df["HOME_TEAM"] == "United States"]
    assert len(unplayed) == 1
    assert pd.isna(unplayed["HOME_GOALS"].iloc[0])


def test_when_score_is_na_string_in_csv_then_away_goals_becomes_pandas_na(loader: Martj42Loader):
    df = loader.load_results()
    unplayed = df[df["HOME_TEAM"] == "United States"]
    assert pd.isna(unplayed["AWAY_GOALS"].iloc[0])


def test_when_results_loaded_then_neutral_column_dtype_is_bool(loader: Martj42Loader):
    df = loader.load_results()
    # dtype.kind == 'b' covers both numpy bool and pandas BooleanDtype
    assert df["NEUTRAL"].dtype.kind == "b"


def test_when_neutral_string_is_true_in_csv_then_neutral_value_is_python_true(
    loader: Martj42Loader,
):
    df = loader.load_results()
    # France vs Croatia — neutral == "TRUE" in the fixture CSV
    france_row = df[df["HOME_TEAM"] == "France"]
    assert france_row["NEUTRAL"].iloc[0] == True  # noqa: E712


def test_when_neutral_string_is_false_in_csv_then_neutral_value_is_python_false(
    loader: Martj42Loader,
):
    df = loader.load_results()
    # England vs Albania — neutral == "FALSE" in the fixture CSV
    england_row = df[df["HOME_TEAM"] == "England"]
    assert england_row["NEUTRAL"].iloc[0] == False  # noqa: E712


# =============================================================================
# Criterion 2 — load_shootouts() and load_goalscorers() schemas
# =============================================================================


def test_when_shootouts_loaded_then_exact_columns_are_returned(loader: Martj42Loader):
    df = loader.load_shootouts()
    assert set(df.columns) == {"DATE", "HOME_TEAM", "AWAY_TEAM", "WINNER", "FIRST_SHOOTER"}


def test_when_goalscorers_loaded_then_exact_columns_are_returned(loader: Martj42Loader):
    df = loader.load_goalscorers()
    assert set(df.columns) == {
        "DATE",
        "HOME_TEAM",
        "AWAY_TEAM",
        "TEAM",
        "SCORER",
        "MINUTE",
        "OWN_GOAL",
        "PENALTY",
    }


def test_when_goalscorers_loaded_then_own_goal_dtype_is_bool(loader: Martj42Loader):
    df = loader.load_goalscorers()
    assert df["OWN_GOAL"].dtype.kind == "b"


def test_when_goalscorers_loaded_then_penalty_dtype_is_bool(loader: Martj42Loader):
    df = loader.load_goalscorers()
    assert df["PENALTY"].dtype.kind == "b"


def test_when_shootouts_loaded_then_row_count_matches_fixture(loader: Martj42Loader):
    df = loader.load_shootouts()
    assert len(df) == 2


def test_when_goalscorers_loaded_then_row_count_matches_fixture(loader: Martj42Loader):
    df = loader.load_goalscorers()
    assert len(df) == 3


# =============================================================================
# Criterion 3 — team-name columns pass through normalize_team / normalize_series
# =============================================================================


def test_when_results_loaded_then_ir_iran_is_normalized_to_iran_in_home_team(loader: Martj42Loader):
    df = loader.load_results()
    assert "IR Iran" not in df["HOME_TEAM"].values
    assert "Iran" in df["HOME_TEAM"].values


def test_when_results_loaded_then_home_team_contains_no_known_unnormalized_aliases(
    loader: Martj42Loader,
):
    from worldcup_playoff.data.crosswalk import CANONICAL_NAMES

    # Aliases whose canonical form differs from the alias itself
    unnormalized = {k for k, v in CANONICAL_NAMES.items() if k != v}
    df = loader.load_results()
    present = set(df["HOME_TEAM"].values) | set(df["AWAY_TEAM"].values)
    collisions = unnormalized & present
    assert not collisions, f"Unnormalized aliases still present in results: {collisions}"


def test_when_goalscorers_loaded_then_team_column_contains_no_unnormalized_aliases(
    loader: Martj42Loader,
):
    from worldcup_playoff.data.crosswalk import CANONICAL_NAMES

    unnormalized = {k for k, v in CANONICAL_NAMES.items() if k != v}
    df = loader.load_goalscorers()
    collisions = unnormalized & set(df["TEAM"].values)
    assert not collisions, f"Unnormalized aliases still present in TEAM column: {collisions}"


def test_when_goalscorers_loaded_then_ir_iran_team_is_normalized_to_iran(loader: Martj42Loader):
    df = loader.load_goalscorers()
    assert "IR Iran" not in df["TEAM"].values


# =============================================================================
# Criterion 4 — REQUIRED_MARTJ42_*_COLUMNS dicts and validator wrappers
# =============================================================================


def test_when_required_results_columns_is_accessed_then_it_is_a_dict():
    assert isinstance(REQUIRED_MARTJ42_RESULTS_COLUMNS, dict)


def test_when_required_shootouts_columns_is_accessed_then_it_is_a_dict():
    assert isinstance(REQUIRED_MARTJ42_SHOOTOUTS_COLUMNS, dict)


def test_when_required_goalscorers_columns_is_accessed_then_it_is_a_dict():
    assert isinstance(REQUIRED_MARTJ42_GOALSCORERS_COLUMNS, dict)


def test_when_required_results_columns_keys_are_checked_then_all_contract_columns_are_present():
    expected = {
        "DATE",
        "HOME_TEAM",
        "AWAY_TEAM",
        "HOME_GOALS",
        "AWAY_GOALS",
        "TOURNAMENT",
        "NEUTRAL",
    }
    assert set(REQUIRED_MARTJ42_RESULTS_COLUMNS.keys()) == expected


def test_when_required_shootouts_columns_keys_are_checked_then_all_contract_columns_are_present():
    assert set(REQUIRED_MARTJ42_SHOOTOUTS_COLUMNS.keys()) == {
        "DATE",
        "HOME_TEAM",
        "AWAY_TEAM",
        "WINNER",
        "FIRST_SHOOTER",
    }


def test_when_required_goalscorers_columns_keys_are_checked_then_all_contract_columns_are_present():
    assert set(REQUIRED_MARTJ42_GOALSCORERS_COLUMNS.keys()) == {
        "DATE",
        "HOME_TEAM",
        "AWAY_TEAM",
        "TEAM",
        "SCORER",
        "MINUTE",
        "OWN_GOAL",
        "PENALTY",
    }


# --- validator: valid frames pass silently ---


def test_when_valid_results_df_is_validated_then_no_error_is_raised():
    validate_results_df(_make_valid_results_df())


def test_when_valid_shootouts_df_is_validated_then_no_error_is_raised():
    validate_shootouts_df(_make_valid_shootouts_df())


def test_when_valid_goalscorers_df_is_validated_then_no_error_is_raised():
    validate_goalscorers_df(_make_valid_goalscorers_df())


def test_when_results_df_has_int64_goals_then_validator_accepts_int64_dtype():
    """Criterion explicitly states the validator must accept Int64."""
    df = _make_valid_results_df()
    assert df["HOME_GOALS"].dtype == pd.Int64Dtype()
    validate_results_df(df)  # must not raise


def test_when_results_df_has_bool_neutral_then_validator_accepts_bool_dtype():
    """Criterion explicitly states the validator must accept bool."""
    df = _make_valid_results_df()
    assert df["NEUTRAL"].dtype.kind == "b"
    validate_results_df(df)  # must not raise


# --- validator: missing column → ValueError ---


def test_when_results_df_is_missing_home_goals_then_value_error_is_raised():
    df = _make_valid_results_df().drop(columns=["HOME_GOALS"])
    with pytest.raises(ValueError):
        validate_results_df(df)


def test_when_results_df_is_missing_neutral_then_value_error_is_raised():
    df = _make_valid_results_df().drop(columns=["NEUTRAL"])
    with pytest.raises(ValueError):
        validate_results_df(df)


def test_when_shootouts_df_is_missing_winner_then_value_error_is_raised():
    df = _make_valid_shootouts_df().drop(columns=["WINNER"])
    with pytest.raises(ValueError):
        validate_shootouts_df(df)


def test_when_goalscorers_df_is_missing_scorer_then_value_error_is_raised():
    df = _make_valid_goalscorers_df().drop(columns=["SCORER"])
    with pytest.raises(ValueError):
        validate_goalscorers_df(df)


# --- validator: wrong dtype → ValueError ---


def test_when_results_df_has_float_home_goals_then_value_error_is_raised():
    # float64 is not the declared Int64 dtype; validator must reject it
    df = _make_valid_results_df().copy()
    df["HOME_GOALS"] = df["HOME_GOALS"].astype(float)
    with pytest.raises(ValueError):
        validate_results_df(df)


def test_when_results_df_has_string_neutral_then_value_error_is_raised():
    # object/string dtype is not bool; validator must reject it
    df = _make_valid_results_df().copy()
    df["NEUTRAL"] = df["NEUTRAL"].astype(str)
    with pytest.raises(ValueError):
        validate_results_df(df)


# =============================================================================
# Criterion 5 — Cache-first: no HTTP when cache files exist; Martj42Config on AppConfig
# =============================================================================


def test_when_martj42_config_is_imported_then_the_class_exists():
    from worldcup_playoff.config import Martj42Config as MC  # noqa: F401


def test_when_martj42_config_is_constructed_with_cache_dir_then_no_error_is_raised(
    tmp_path: Path,
):
    Martj42Config(cache_dir=tmp_path)


def test_when_app_config_is_inspected_then_martj42_field_is_present():
    # Pydantic v2 exposes model_fields; v1 exposes __fields__
    fields = getattr(AppConfig, "model_fields", None) or getattr(AppConfig, "__fields__", {})
    assert "martj42" in fields, "AppConfig must expose a 'martj42' field (Martj42Config)"


def test_when_cache_files_exist_then_load_results_does_not_call_requests_get(cache_dir: Path):
    """Cache-first contract: if results.csv is in cache_dir, no HTTP request is made."""
    ldr = Martj42Loader(config=Martj42Config(cache_dir=cache_dir))
    with patch("requests.get", side_effect=AssertionError("requests.get must not be called")) as m:
        ldr.load_results()
    m.assert_not_called()


def test_when_cache_files_exist_then_load_shootouts_does_not_call_requests_get(cache_dir: Path):
    ldr = Martj42Loader(config=Martj42Config(cache_dir=cache_dir))
    with patch("requests.get", side_effect=AssertionError("requests.get must not be called")) as m:
        ldr.load_shootouts()
    m.assert_not_called()


def test_when_cache_files_exist_then_load_goalscorers_does_not_call_requests_get(cache_dir: Path):
    ldr = Martj42Loader(config=Martj42Config(cache_dir=cache_dir))
    with patch("requests.get", side_effect=AssertionError("requests.get must not be called")) as m:
        ldr.load_goalscorers()
    m.assert_not_called()


# =============================================================================
# Criterion 6 — wc2026_schedule() convenience filter
# =============================================================================


def test_when_wc2026_schedule_is_called_then_result_is_a_dataframe(loader: Martj42Loader):
    assert isinstance(loader.wc2026_schedule(), pd.DataFrame)


def test_when_wc2026_schedule_is_called_then_only_fifa_world_cup_rows_are_returned(
    loader: Martj42Loader,
):
    df = loader.wc2026_schedule()
    assert (df["TOURNAMENT"] == "FIFA World Cup").all()


def test_when_wc2026_schedule_is_called_then_non_world_cup_rows_are_excluded(
    loader: Martj42Loader,
):
    # "UEFA Nations League" row in fixture data must not appear in the schedule
    df = loader.wc2026_schedule()
    assert "UEFA Nations League" not in df["TOURNAMENT"].values


def test_when_wc2026_schedule_is_called_then_unplayed_na_score_fixtures_are_included(
    loader: Martj42Loader,
):
    # The 2026 US vs Mexico fixture has NA scores — it must be present
    df = loader.wc2026_schedule()
    assert df["HOME_GOALS"].isna().any(), "unplayed (<NA>-score) fixtures must be included"


def test_when_wc2026_schedule_is_called_then_played_fixtures_are_also_included(
    loader: Martj42Loader,
):
    df = loader.wc2026_schedule()
    played = df.dropna(subset=["HOME_GOALS", "AWAY_GOALS"])
    assert len(played) > 0


def test_when_wc2026_schedule_is_called_then_pre_2026_world_cup_rows_are_excluded(
    loader: Martj42Loader,
):
    # The fixture contains 2006 and 2018 FIFA World Cup rows; they must not appear.
    df = loader.wc2026_schedule()
    if len(df) > 0:
        years = pd.to_datetime(df["DATE"]).dt.year
        assert (years == 2026).all(), "Non-2026 World Cup rows must be filtered out"


# =============================================================================
# Criterion 7 — schema tests use mocked network (fixture CSV in tmp_path)
# — verified structurally: all tests in this file use cache_dir (tmp_path),
#   no real HTTP is triggered anywhere above.
# =============================================================================


def test_when_results_loaded_via_cache_then_row_count_matches_fixture(loader: Martj42Loader):
    df = loader.load_results()
    assert len(df) == 6  # six rows in _RESULTS_CSV


def test_when_results_loaded_via_cache_then_german_goal_tally_is_correct(loader: Martj42Loader):
    df = loader.load_results()
    row = df[df["HOME_TEAM"] == "Germany"]
    assert row["HOME_GOALS"].iloc[0] == 4


# =============================================================================
# Property-based tests (Hypothesis)
# =============================================================================


# --- Filter invariant: wc2026_schedule(df) only ever returns FIFA World Cup rows ---

_TOURNAMENT_NAMES = st.sampled_from(
    [
        "FIFA World Cup",
        "UEFA Nations League",
        "Friendly",
        "CONMEBOL World Cup Qualifying",
        "AFC Cup",
    ]
)


@given(
    st.lists(
        st.fixed_dictionaries(
            {
                "DATE": st.just("2026-06-11"),
                "HOME_TEAM": st.text(min_size=1, max_size=30),
                "AWAY_TEAM": st.text(min_size=1, max_size=30),
                "HOME_GOALS": st.none(),
                "AWAY_GOALS": st.none(),
                "TOURNAMENT": _TOURNAMENT_NAMES,
                "NEUTRAL": st.booleans(),
            }
        ),
        min_size=0,
        max_size=30,
    )
)
def test_when_wc2026_schedule_filters_any_results_df_then_all_rows_are_world_cup(
    rows: list,
):
    """Filter invariant: wc2026_schedule never admits non-FIFA-World-Cup rows."""
    df = (
        pd.DataFrame(rows)
        if rows
        else pd.DataFrame(
            columns=[
                "DATE",
                "HOME_TEAM",
                "AWAY_TEAM",
                "HOME_GOALS",
                "AWAY_GOALS",
                "TOURNAMENT",
                "NEUTRAL",
            ]
        )
    )
    result = wc2026_schedule(df)
    assert isinstance(result, pd.DataFrame)
    if len(result) > 0:
        assert (result["TOURNAMENT"] == "FIFA World Cup").all()


# --- Missing-column invariant: validate_results_df always raises for any absent required column ---

_RESULTS_REQUIRED_COLS = [
    "DATE",
    "HOME_TEAM",
    "AWAY_TEAM",
    "HOME_GOALS",
    "AWAY_GOALS",
    "TOURNAMENT",
    "NEUTRAL",
]


@given(st.sampled_from(_RESULTS_REQUIRED_COLS))
def test_when_any_required_results_column_is_absent_then_value_error_is_raised(missing: str):
    """Missing-column invariant for results validator."""
    df = _make_valid_results_df().drop(columns=[missing])
    with pytest.raises(ValueError):
        validate_results_df(df)


_SHOOTOUTS_REQUIRED_COLS = ["DATE", "HOME_TEAM", "AWAY_TEAM", "WINNER", "FIRST_SHOOTER"]


@given(st.sampled_from(_SHOOTOUTS_REQUIRED_COLS))
def test_when_any_required_shootouts_column_is_absent_then_value_error_is_raised(missing: str):
    """Missing-column invariant for shootouts validator."""
    df = _make_valid_shootouts_df().drop(columns=[missing])
    with pytest.raises(ValueError):
        validate_shootouts_df(df)


_GOALSCORERS_REQUIRED_COLS = [
    "DATE",
    "HOME_TEAM",
    "AWAY_TEAM",
    "TEAM",
    "SCORER",
    "MINUTE",
    "OWN_GOAL",
    "PENALTY",
]


@given(st.sampled_from(_GOALSCORERS_REQUIRED_COLS))
def test_when_any_required_goalscorers_column_is_absent_then_value_error_is_raised(missing: str):
    """Missing-column invariant for goalscorers validator."""
    df = _make_valid_goalscorers_df().drop(columns=[missing])
    with pytest.raises(ValueError):
        validate_goalscorers_df(df)
