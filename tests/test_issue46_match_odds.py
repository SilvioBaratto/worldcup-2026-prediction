"""
Source-blind example tests for worldcup_playoff.data.odds match odds functions.
Issue #46: feat: archived bookmaker match-odds scraper + de-vig → WDL probabilities + CSV cache.

Tests derived exclusively from acceptance criteria — no implementation source was read (TDD Red).
All five runtime-verifiable criteria are covered; the two non-verifiable criteria (suite gate,
code-quality prose) are intentionally skipped per the oracle report.

Fixture: tests/fixtures/match_odds_sample.html — a synthetic archived 1X2 odds page for WC2018
group-stage matches, designed to exercise team-name normalization (IR Iran) and numeric parsing.

Config assumption: load_match_odds(tournament, config) reads config.odds.cache_dir for the CSV
cache location, mirroring the [odds] TOML section described in requirements.md §10.

Internal fetch assumption: the module exposes a private _fetch_match_html(url) function that
load_match_odds calls for network access; tests patch this to avoid live network traffic.
"""

from __future__ import annotations

import math
import pathlib
import types
from unittest.mock import patch

import pandas as pd
import pytest
from hypothesis import given, strategies as st

from worldcup_playoff.data.odds import load_match_odds, parse_match_html, to_match_probs

FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures"

_PATCH_TARGET = "worldcup_playoff.data.odds._fetch_match_html"


def _match_html() -> str:
    return (FIXTURES_DIR / "match_odds_sample.html").read_text(encoding="utf-8")


def _make_config(tmp_path: pathlib.Path) -> types.SimpleNamespace:
    """Minimal config stub that provides an odds.cache_dir path."""
    return types.SimpleNamespace(odds=types.SimpleNamespace(cache_dir=tmp_path))


# ---------------------------------------------------------------------------
# Criterion 1: parse_match_html returns rows of date, home_team, away_team,
#              o_home, o_draw, o_away
# ---------------------------------------------------------------------------


def test_when_match_html_parsed_then_result_is_a_dataframe():
    result = parse_match_html(_match_html())
    assert isinstance(result, pd.DataFrame)


def test_when_match_html_parsed_then_result_is_non_empty():
    result = parse_match_html(_match_html())
    assert not result.empty


def test_when_match_html_parsed_then_required_columns_are_present():
    """Criterion 1: all six output columns must be present."""
    required = {"date", "home_team", "away_team", "o_home", "o_draw", "o_away"}
    result = parse_match_html(_match_html())
    missing = required - set(result.columns)
    assert not missing, f"Missing columns: {missing}"


def test_when_match_html_parsed_then_odds_columns_are_numeric():
    result = parse_match_html(_match_html())
    for col in ("o_home", "o_draw", "o_away"):
        assert pd.api.types.is_numeric_dtype(result[col]), f"{col!r} is not numeric"


def test_when_match_html_parsed_then_all_odds_are_strictly_positive():
    result = parse_match_html(_match_html())
    for col in ("o_home", "o_draw", "o_away"):
        assert (result[col] > 0).all(), f"Non-positive odds found in {col!r}"


def test_when_match_html_parsed_then_home_team_column_contains_non_empty_strings():
    result = parse_match_html(_match_html())
    assert result["home_team"].apply(lambda v: isinstance(v, str) and len(v) > 0).all()


def test_when_match_html_parsed_then_away_team_column_contains_non_empty_strings():
    result = parse_match_html(_match_html())
    assert result["away_team"].apply(lambda v: isinstance(v, str) and len(v) > 0).all()


def test_when_match_html_parsed_then_fixture_home_team_appears():
    """Russia is the first home team in the fixture."""
    result = parse_match_html(_match_html())
    assert "Russia" in result["home_team"].values


def test_when_match_html_parsed_then_fixture_away_team_appears():
    """Saudi Arabia is the first away team in the fixture."""
    result = parse_match_html(_match_html())
    assert "Saudi Arabia" in result["away_team"].values


def test_when_match_html_parsed_then_row_count_matches_fixture_matches():
    """Fixture HTML contains exactly 5 match rows."""
    result = parse_match_html(_match_html())
    assert len(result) == 5


def test_when_match_html_parsed_then_date_column_is_present():
    result = parse_match_html(_match_html())
    assert "date" in result.columns


# ---------------------------------------------------------------------------
# Criterion 2: to_match_probs de-vigs each row to p_win, p_draw, p_loss
#              where each row sums to ≈ 1.0
# ---------------------------------------------------------------------------


def test_when_to_match_probs_called_then_output_has_p_win_p_draw_p_loss_columns():
    df = parse_match_html(_match_html())
    result = to_match_probs(df)
    missing = {"p_win", "p_draw", "p_loss"} - set(result.columns)
    assert not missing, f"Missing probability columns: {missing}"


def test_when_to_match_probs_called_then_each_row_sums_to_approx_one():
    """Criterion 2 core invariant: de-vig → probabilities sum to 1.0 per row."""
    df = parse_match_html(_match_html())
    result = to_match_probs(df)
    row_sums = result["p_win"] + result["p_draw"] + result["p_loss"]
    assert (abs(row_sums - 1.0) < 1e-9).all(), f"Row sums deviate from 1.0: {row_sums.tolist()}"


def test_when_to_match_probs_called_then_all_probs_are_strictly_positive():
    df = parse_match_html(_match_html())
    result = to_match_probs(df)
    for col in ("p_win", "p_draw", "p_loss"):
        assert (result[col] > 0).all(), f"Non-positive probabilities in {col!r}"


def test_when_to_match_probs_called_on_known_row_then_values_match_de_vig_formula():
    """De-vig formula: p_i = (1/o_i) / Σ(1/o_j). Verify against a hand-computed row."""
    o_home, o_draw, o_away = 1.45, 4.20, 7.50
    inv_sum = (1 / o_home) + (1 / o_draw) + (1 / o_away)
    expected = {
        "p_win": (1 / o_home) / inv_sum,
        "p_draw": (1 / o_draw) / inv_sum,
        "p_loss": (1 / o_away) / inv_sum,
    }
    df = pd.DataFrame([{"o_home": o_home, "o_draw": o_draw, "o_away": o_away}])
    result = to_match_probs(df)
    for col, exp in expected.items():
        assert math.isclose(result[col].iloc[0], exp, rel_tol=1e-9), (
            f"{col}: expected {exp}, got {result[col].iloc[0]}"
        )


def test_when_to_match_probs_called_then_row_count_is_preserved():
    df = parse_match_html(_match_html())
    result = to_match_probs(df)
    assert len(result) == len(df)


@given(
    st.lists(
        st.tuples(
            st.floats(min_value=1.01, max_value=100.0, allow_nan=False, allow_infinity=False),
            st.floats(min_value=1.01, max_value=100.0, allow_nan=False, allow_infinity=False),
            st.floats(min_value=1.01, max_value=100.0, allow_nan=False, allow_infinity=False),
        ),
        min_size=1,
        max_size=20,
    )
)
def test_when_any_valid_1x2_odds_de_vigged_then_each_row_sums_to_one(rows):
    """Invariant: to_match_probs rows always sum to 1.0 for any valid 1X2 decimal odds."""
    df = pd.DataFrame(rows, columns=["o_home", "o_draw", "o_away"])
    result = to_match_probs(df)
    for s in result["p_win"] + result["p_draw"] + result["p_loss"]:
        assert math.isclose(s, 1.0, abs_tol=1e-9), f"Row sum was {s}, expected 1.0"


@given(
    st.lists(
        st.tuples(
            st.floats(min_value=1.01, max_value=100.0, allow_nan=False, allow_infinity=False),
            st.floats(min_value=1.01, max_value=100.0, allow_nan=False, allow_infinity=False),
            st.floats(min_value=1.01, max_value=100.0, allow_nan=False, allow_infinity=False),
        ),
        min_size=1,
        max_size=20,
    )
)
def test_when_any_valid_1x2_odds_de_vigged_then_all_probs_are_positive(rows):
    """Invariant: every de-vigged probability is strictly positive for valid input odds."""
    df = pd.DataFrame(rows, columns=["o_home", "o_draw", "o_away"])
    result = to_match_probs(df)
    assert (result["p_win"] > 0).all()
    assert (result["p_draw"] > 0).all()
    assert (result["p_loss"] > 0).all()


# ---------------------------------------------------------------------------
# Criterion 3: load_match_odds is cache-first; empty frame on block, never raises
# ---------------------------------------------------------------------------


def test_when_load_match_odds_called_first_time_then_csv_is_written_to_cache(tmp_path):
    config = _make_config(tmp_path)
    with patch(_PATCH_TARGET, return_value=_match_html()):
        load_match_odds("wc2018", config)
    assert list(tmp_path.glob("*.csv")), "No CSV was written to cache_dir after first load"


def test_when_load_match_odds_called_first_time_then_result_is_a_dataframe(tmp_path):
    config = _make_config(tmp_path)
    with patch(_PATCH_TARGET, return_value=_match_html()):
        result = load_match_odds("wc2018", config)
    assert isinstance(result, pd.DataFrame)


def test_when_load_match_odds_called_second_time_then_fetch_is_not_called(tmp_path):
    """Cache-first: second call must read from CSV and must not call the network fetch."""
    config = _make_config(tmp_path)
    with patch(_PATCH_TARGET, return_value=_match_html()) as mock_fetch:
        load_match_odds("wc2018", config)  # populates cache
        mock_fetch.reset_mock()
        result = load_match_odds("wc2018", config)  # must use cache
    assert mock_fetch.call_count == 0, "Network fetch was called on second load despite cache hit"
    assert isinstance(result, pd.DataFrame)
    assert not result.empty


def test_when_cache_exists_then_cached_and_fresh_loads_have_same_shape(tmp_path):
    config = _make_config(tmp_path)
    with patch(_PATCH_TARGET, return_value=_match_html()):
        first = load_match_odds("wc2018", config)
    second = load_match_odds("wc2018", config)  # cache is now populated; no patch needed
    assert first.shape == second.shape
    assert list(first.columns) == list(second.columns)


def test_when_fetch_raises_oserror_then_load_match_odds_returns_empty_frame(tmp_path):
    """Graceful degradation: OSError → empty DataFrame, not a raised exception."""
    config = _make_config(tmp_path)
    with patch(_PATCH_TARGET, side_effect=OSError("503 Service Unavailable")):
        result = load_match_odds("wc2022", config)
    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_when_fetch_raises_connection_error_then_load_match_odds_returns_empty_frame(tmp_path):
    """Graceful degradation: 403 / ConnectionError → empty DataFrame, not a raised exception."""
    config = _make_config(tmp_path)
    with patch(_PATCH_TARGET, side_effect=ConnectionError("403 Forbidden")):
        result = load_match_odds("wc2014", config)
    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_when_fetch_raises_runtime_error_then_load_match_odds_does_not_propagate(tmp_path):
    """Any exception from the fetch layer must be caught; load_match_odds must never raise."""
    config = _make_config(tmp_path)
    with patch(_PATCH_TARGET, side_effect=RuntimeError("Unexpected scraper failure")):
        try:
            load_match_odds("wc2018", config)
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"load_match_odds() propagated an exception to the caller: {exc!r}")


# ---------------------------------------------------------------------------
# Criterion 4: output keyed date/home_team/away_team; team names normalized
#              via crosswalk.normalize_team
# ---------------------------------------------------------------------------


def test_when_match_odds_loaded_then_output_has_date_key(tmp_path):
    """'date' must be a column or an index level in the output frame."""
    config = _make_config(tmp_path)
    with patch(_PATCH_TARGET, return_value=_match_html()):
        result = load_match_odds("wc2018", config)
    assert "date" in result.columns or "date" in result.index.names


def test_when_match_odds_loaded_then_output_has_home_team_key(tmp_path):
    config = _make_config(tmp_path)
    with patch(_PATCH_TARGET, return_value=_match_html()):
        result = load_match_odds("wc2018", config)
    assert "home_team" in result.columns or "home_team" in result.index.names


def test_when_match_odds_loaded_then_output_has_away_team_key(tmp_path):
    config = _make_config(tmp_path)
    with patch(_PATCH_TARGET, return_value=_match_html()):
        result = load_match_odds("wc2018", config)
    assert "away_team" in result.columns or "away_team" in result.index.names


def test_when_match_odds_loaded_then_team_names_are_normalized_via_crosswalk(tmp_path, monkeypatch):
    """Criterion 4: load_match_odds must call crosswalk.normalize_team for team name columns."""
    from worldcup_playoff.data import crosswalk

    config = _make_config(tmp_path)
    calls: list[str] = []
    original = crosswalk.normalize_team

    def _spy(name: str) -> str:
        calls.append(name)
        return original(name)

    monkeypatch.setattr(crosswalk, "normalize_team", _spy)
    with patch(_PATCH_TARGET, return_value=_match_html()):
        load_match_odds("wc2018", config)

    assert len(calls) > 0, "crosswalk.normalize_team was never called during load_match_odds()"


def test_when_match_odds_loaded_then_ir_iran_is_normalized(tmp_path):
    """Fixture contains 'IR Iran'; the output must show the canonical form 'Iran'."""
    config = _make_config(tmp_path)
    with patch(_PATCH_TARGET, return_value=_match_html()):
        result = load_match_odds("wc2018", config)

    def _flatten_teams(df: pd.DataFrame) -> set[str]:
        teams: set[str] = set()
        if "home_team" in df.columns:
            teams |= set(df["home_team"])
        elif "home_team" in df.index.names:
            teams |= set(df.index.get_level_values("home_team"))
        if "away_team" in df.columns:
            teams |= set(df["away_team"])
        elif "away_team" in df.index.names:
            teams |= set(df.index.get_level_values("away_team"))
        return teams

    teams = _flatten_teams(result)
    assert "IR Iran" not in teams, "'IR Iran' was not normalized (expected 'Iran')"
    assert "Iran" in teams, "'Iran' (canonical form) not found in output teams"


def test_when_match_odds_loaded_then_date_home_away_uniquely_identify_rows(tmp_path):
    """Key criterion: the combination date + home_team + away_team must be unique per row."""
    config = _make_config(tmp_path)
    with patch(_PATCH_TARGET, return_value=_match_html()):
        result = load_match_odds("wc2018", config)

    def _key_cols(df: pd.DataFrame) -> pd.DataFrame:
        cols = ["date", "home_team", "away_team"]
        available = [c for c in cols if c in df.columns]
        return df[available] if available else df.index.to_frame(index=False)[cols]

    keys = _key_cols(result)
    assert not keys.duplicated().any(), "date/home_team/away_team combination is not unique"
