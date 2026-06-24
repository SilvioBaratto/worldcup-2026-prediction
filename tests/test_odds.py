"""
Tests for worldcup_playoff.data.odds — archived bookmaker odds scraper + de-vig + CSV cache.
Source-blind: written against acceptance criteria before any implementation (TDD Red phase).
Issue #20: feat: archived bookmaker odds scraper + de-vig + CSV cache (data/odds.py)
"""

from __future__ import annotations

import math
import pathlib

import pandas as pd
import pytest
from hypothesis import given, strategies as st

from worldcup_playoff.data.odds import OddsScraper, de_vig

FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures"


def _html() -> str:
    return (FIXTURES_DIR / "odds_sample.html").read_text(encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Criterion 2: de-vig formula — p_i = (1/o_i) / Σ(1/o_j)
# ─────────────────────────────────────────────────────────────────────────────


def test_when_two_way_market_de_vigged_then_probs_sum_to_one():
    assert math.isclose(sum(de_vig([1.91, 1.91])), 1.0, abs_tol=1e-9)


def test_when_three_way_market_de_vigged_then_each_prob_matches_formula():
    odds = [2.5, 3.2, 4.0]
    probs = de_vig(odds)
    inv_sum = sum(1 / o for o in odds)
    for prob, odd in zip(probs, odds):
        assert math.isclose(prob, (1 / odd) / inv_sum, rel_tol=1e-9)


def test_when_de_vigged_then_output_length_matches_input_length():
    odds = [2.0, 3.0, 5.0, 10.0]
    assert len(de_vig(odds)) == len(odds)


@given(
    st.lists(
        st.floats(min_value=1.01, max_value=200.0, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=30,
    )
)
def test_when_any_valid_decimal_odds_de_vigged_then_probs_sum_to_one(odds):
    """Invariant: de_vig output always sums to 1.0 for any valid decimal-odds list."""
    assert math.isclose(sum(de_vig(odds)), 1.0, abs_tol=1e-9)


@given(
    st.lists(
        st.floats(min_value=1.01, max_value=200.0, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=30,
    )
)
def test_when_any_valid_decimal_odds_de_vigged_then_all_probs_are_positive(odds):
    """Invariant: every de-vigged probability is strictly positive."""
    assert all(p > 0 for p in de_vig(odds))


@given(
    st.lists(
        st.floats(min_value=1.01, max_value=200.0, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=30,
    )
)
def test_when_any_valid_decimal_odds_given_then_de_vig_does_not_raise(odds):
    """Invariant: de_vig never raises for any list of valid decimal odds (≥ 1.01)."""
    de_vig(odds)  # must not raise


# ─────────────────────────────────────────────────────────────────────────────
# Criterion 1: BeautifulSoup HTML parsing (saved fixture, zero live network)
# ─────────────────────────────────────────────────────────────────────────────


def test_when_html_fixture_given_then_parse_html_returns_dataframe(tmp_path):
    scraper = OddsScraper(cache_dir=tmp_path)
    result = scraper.parse_html(_html())
    assert isinstance(result, pd.DataFrame)
    assert not result.empty


def test_when_html_fixture_parsed_then_team_column_is_present(tmp_path):
    scraper = OddsScraper(cache_dir=tmp_path)
    df = scraper.parse_html(_html())
    assert "team" in df.columns


def test_when_html_fixture_parsed_then_odds_column_is_present(tmp_path):
    scraper = OddsScraper(cache_dir=tmp_path)
    df = scraper.parse_html(_html())
    assert "odds" in df.columns


def test_when_html_fixture_parsed_then_odds_are_numeric_and_positive(tmp_path):
    scraper = OddsScraper(cache_dir=tmp_path)
    df = scraper.parse_html(_html())
    assert pd.api.types.is_numeric_dtype(df["odds"])
    assert (df["odds"] > 0).all()


def test_when_html_fixture_parsed_then_fixture_team_appears_in_result(tmp_path):
    """'Germany' is in the fixture HTML; it must appear in the parsed output."""
    scraper = OddsScraper(cache_dir=tmp_path)
    df = scraper.parse_html(_html())
    assert any("Germany" in str(t) for t in df["team"])


# ─────────────────────────────────────────────────────────────────────────────
# Criterion 3: cache-first — write on first fetch, read from CSV on subsequent
# ─────────────────────────────────────────────────────────────────────────────


def test_when_no_cache_exists_then_csv_is_written_to_cache_dir_after_load(tmp_path):
    scraper = OddsScraper(cache_dir=tmp_path)
    scraper._fetch_html = lambda url: _html()
    scraper.load("wc2018")
    assert list(tmp_path.glob("*.csv")), "No CSV was written to cache_dir"


def test_when_cache_populated_then_second_load_does_not_call_fetch_html(tmp_path):
    """After the first load populates the cache, a second load must not call _fetch_html."""
    scraper = OddsScraper(cache_dir=tmp_path)
    scraper._fetch_html = lambda url: _html()
    scraper.load("wc2018")  # populates cache

    call_log: list[str] = []

    def _spy(url: str) -> str:
        call_log.append(url)
        return _html()

    scraper2 = OddsScraper(cache_dir=tmp_path)
    scraper2._fetch_html = _spy
    result = scraper2.load("wc2018")

    assert not result.empty
    assert call_log == [], "Network was called despite a valid CSV cache existing"


def test_when_cache_populated_then_second_load_returns_same_shape(tmp_path):
    """Cached and fresh loads produce DataFrames with identical shapes and columns."""
    scraper = OddsScraper(cache_dir=tmp_path)
    scraper._fetch_html = lambda url: _html()
    first = scraper.load("wc2018")

    scraper2 = OddsScraper(cache_dir=tmp_path)
    scraper2._fetch_html = lambda url: _html()
    second = scraper2.load("wc2018")

    assert first.shape == second.shape
    assert list(first.columns) == list(second.columns)


# ─────────────────────────────────────────────────────────────────────────────
# Criterion 7 (sub-test): simulated 403 / exception → empty frame, never raises
# ─────────────────────────────────────────────────────────────────────────────


def test_when_fetch_raises_connection_error_then_empty_frame_is_returned(tmp_path):
    """Simulated 403 — ConnectionError during fetch must yield empty DataFrame, not an exception."""

    def _raise(url: str) -> str:
        raise ConnectionError("403 Forbidden")

    scraper = OddsScraper(cache_dir=tmp_path)
    scraper._fetch_html = _raise
    result = scraper.load("wc2022")
    assert isinstance(result, pd.DataFrame)


def test_when_fetch_raises_oserror_then_empty_frame_is_returned(tmp_path):
    """Simulated blocked source — OSError during fetch must yield empty DataFrame, not an exception."""

    def _raise(url: str) -> str:
        raise OSError("503 Service Unavailable")

    scraper = OddsScraper(cache_dir=tmp_path)
    scraper._fetch_html = _raise
    result = scraper.load("wc2014")
    assert isinstance(result, pd.DataFrame)


def test_when_fetch_raises_any_exception_then_load_does_not_propagate_it(tmp_path):
    """Any exception raised by _fetch_html must be caught; load() must never propagate it."""

    def _raise(url: str) -> str:
        raise RuntimeError("Unexpected scraper failure")

    scraper = OddsScraper(cache_dir=tmp_path)
    scraper._fetch_html = _raise
    try:
        scraper.load("wc2018")
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"load() propagated an exception to the caller: {exc!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Criterion 5: team names normalized via crosswalk.normalize_team
# ─────────────────────────────────────────────────────────────────────────────


def test_when_odds_loaded_then_crosswalk_normalize_team_is_called(tmp_path, monkeypatch):
    """load() must invoke crosswalk.normalize_team at least once per row in the result."""
    from worldcup_playoff.data import crosswalk

    scraper = OddsScraper(cache_dir=tmp_path)
    scraper._fetch_html = lambda url: _html()

    calls: list[str] = []
    original = crosswalk.normalize_team

    def _tracking(name: str) -> str:
        calls.append(name)
        return original(name)

    monkeypatch.setattr(crosswalk, "normalize_team", _tracking)
    scraper.load("wc2018")

    assert len(calls) > 0, "crosswalk.normalize_team was never called during load()"


# ─────────────────────────────────────────────────────────────────────────────
# Criterion 6: beautifulsoup4 listed in pyproject.toml
# ─────────────────────────────────────────────────────────────────────────────


def test_when_pyproject_toml_read_then_beautifulsoup4_is_listed_as_dependency():
    """beautifulsoup4 must appear in pyproject.toml as a project dependency."""
    pyproject = pathlib.Path(__file__).parent.parent / "pyproject.toml"
    assert pyproject.is_file(), "pyproject.toml not found at project root"
    assert "beautifulsoup4" in pyproject.read_text(encoding="utf-8")
