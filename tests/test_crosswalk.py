"""Source-blind example tests for the team-name crosswalk module (issue #1).

Tests derived exclusively from acceptance criteria — no implementation source was read.
"""

from __future__ import annotations

import pandas as pd
import pytest
from hypothesis import given, strategies as st

from worldcup_playoff.data import CANONICAL_NAMES, normalize_series, normalize_team


# ---------------------------------------------------------------------------
# normalize_team: whitespace handling
# ---------------------------------------------------------------------------


def test_when_name_has_leading_trailing_whitespace_then_it_is_stripped():
    result = normalize_team("  Germany  ")
    assert result == result.strip()


def test_when_name_has_internal_multiple_spaces_then_they_are_collapsed():
    result = normalize_team("United  States")
    assert "  " not in result


def test_when_unknown_name_is_given_then_cleaned_input_is_returned():
    unknown = "Ruritania"
    result = normalize_team(unknown)
    assert result == unknown


def test_when_unknown_name_with_whitespace_is_given_then_stripped_form_is_returned():
    result = normalize_team("  Ruritania  ")
    assert result == "Ruritania"


# ---------------------------------------------------------------------------
# CANONICAL_NAMES alias coverage (data-contract aliases)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "alias,expected_canonical",
    [
        ("Türkiye", "Turkey"),
        ("Turkey", "Turkey"),
        ("Czechia", "Czech Republic"),
        ("South Korea", "South Korea"),
        ("Korea Republic", "South Korea"),
        ("Republic of Korea", "South Korea"),
        ("United States", "United States"),
        ("USA", "United States"),
        ("United States of America", "United States"),
        ("IR Iran", "Iran"),
        ("Iran", "Iran"),
        ("China PR", "China"),
        ("China", "China"),
    ],
)
def test_when_known_alias_is_normalized_then_canonical_name_is_returned(
    alias: str, expected_canonical: str
):
    assert normalize_team(alias) == expected_canonical


def test_when_alias_is_uppercase_then_canonical_is_still_returned():
    assert normalize_team("TÜRKIYE") == "Turkey"


def test_when_alias_is_lowercase_then_canonical_is_still_returned():
    assert normalize_team("türkiye") == "Turkey"


def test_when_alias_has_surrounding_whitespace_then_canonical_is_still_returned():
    assert normalize_team("  IR Iran  ") == "Iran"


def test_when_canonical_names_dict_is_accessed_then_it_is_a_dict():
    assert isinstance(CANONICAL_NAMES, dict)


def test_when_canonical_names_dict_is_checked_then_all_data_contract_aliases_are_covered():
    required_aliases = {
        "türkiye",
        "korea republic",
        "republic of korea",
        "south korea",
        "usa",
        "united states of america",
        "ir iran",
        "china pr",
        "czechia",
    }
    lowered_keys = {k.lower().strip() for k in CANONICAL_NAMES}
    missing = required_aliases - lowered_keys
    assert not missing, f"Missing aliases in CANONICAL_NAMES: {missing}"


# ---------------------------------------------------------------------------
# normalize_series: length and index preservation
# ---------------------------------------------------------------------------


def test_when_series_is_normalized_then_length_is_preserved():
    s = pd.Series(["Germany", "Türkiye", "IR Iran"])
    result = normalize_series(s)
    assert len(result) == len(s)


def test_when_series_is_normalized_then_index_is_preserved():
    s = pd.Series(["Germany", "Türkiye"], index=[10, 20])
    result = normalize_series(s)
    assert list(result.index) == [10, 20]


def test_when_series_contains_known_alias_then_canonical_name_is_returned():
    s = pd.Series(["Türkiye"])
    result = normalize_series(s)
    assert result.iloc[0] == "Turkey"


def test_when_series_contains_unknown_name_then_it_is_returned_cleaned():
    s = pd.Series(["Ruritania"])
    result = normalize_series(s)
    assert result.iloc[0] == "Ruritania"


def test_when_series_is_empty_then_empty_series_is_returned():
    s = pd.Series([], dtype=str)
    result = normalize_series(s)
    assert len(result) == 0


def test_when_series_has_mixed_names_then_all_are_mapped_correctly():
    s = pd.Series(["Türkiye", "Germany", "IR Iran"])
    result = normalize_series(s)
    assert result.tolist() == ["Turkey", "Germany", "Iran"]


# ---------------------------------------------------------------------------
# Idempotence
# ---------------------------------------------------------------------------


def test_when_canonical_name_is_normalized_then_result_is_unchanged():
    for canonical in set(CANONICAL_NAMES.values()):
        assert normalize_team(canonical) == canonical, (
            f"normalize_team({canonical!r}) is not idempotent"
        )


@given(st.text())
def test_when_normalize_team_is_applied_twice_then_result_is_unchanged(name: str):
    """Idempotence invariant: applying normalize_team twice equals applying it once."""
    once = normalize_team(name)
    assert normalize_team(once) == once


# ---------------------------------------------------------------------------
# Public exports from worldcup_playoff/data/__init__.py
# ---------------------------------------------------------------------------


def test_when_normalize_team_is_imported_from_data_package_then_it_is_callable():
    from worldcup_playoff.data import normalize_team as nt

    assert callable(nt)


def test_when_normalize_series_is_imported_from_data_package_then_it_is_callable():
    from worldcup_playoff.data import normalize_series as ns

    assert callable(ns)


def test_when_canonical_names_is_imported_from_data_package_then_it_is_a_dict():
    from worldcup_playoff.data import CANONICAL_NAMES as cn

    assert isinstance(cn, dict)


def test_when_data_package_all_is_checked_then_required_symbols_are_present():
    import worldcup_playoff.data as data_pkg

    all_exports = getattr(data_pkg, "__all__", [])
    for symbol in ("normalize_team", "normalize_series", "CANONICAL_NAMES"):
        assert symbol in all_exports, f"{symbol!r} not found in worldcup_playoff.data.__all__"
