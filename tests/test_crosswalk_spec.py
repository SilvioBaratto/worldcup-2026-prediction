"""Source-blind example tests for issue #26: team-name crosswalk.

Every test is derived solely from the acceptance criteria text. No implementation
source was read during authoring — this is the Red phase of TDD.
"""

from __future__ import annotations

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from worldcup_playoff.data.crosswalk import (
    CANONICAL_NAMES,
    is_known,
    normalize_series,
    normalize_team,
)

# ---------------------------------------------------------------------------
# Criterion: CANONICAL_NAMES maps every explicitly listed divergent spelling
# ---------------------------------------------------------------------------

EXPLICIT_ALIAS_TABLE = [
    ("Türkiye", "Turkey"),
    ("Czechia", "Czech Republic"),
    ("Korea Republic", "South Korea"),
    ("USA", "United States"),
    ("IR Iran", "Iran"),
    ("China PR", "China"),
    ("Côte d'Ivoire", "Ivory Coast"),
    ("Cape Verde", "Cabo Verde"),
]


@pytest.mark.parametrize("alias, canonical", EXPLICIT_ALIAS_TABLE)
def test_when_canonical_names_contains_alias_then_it_maps_to_canonical_form(
    alias: str, canonical: str
) -> None:
    assert CANONICAL_NAMES[alias] == canonical


# ---------------------------------------------------------------------------
# Criterion: normalize_team returns canonical name for known aliases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alias, canonical", EXPLICIT_ALIAS_TABLE)
def test_when_normalize_team_called_with_known_alias_then_canonical_is_returned(
    alias: str, canonical: str
) -> None:
    assert normalize_team(alias) == canonical


# ---------------------------------------------------------------------------
# Criterion: normalize_team returns whitespace-cleaned input when unknown
# ---------------------------------------------------------------------------


def test_when_normalize_team_called_with_unknown_name_then_cleaned_input_is_returned() -> None:
    unknown = "Ruritania"
    assert normalize_team(unknown) == unknown


def test_when_normalize_team_called_with_padded_unknown_name_then_whitespace_is_collapsed() -> None:
    # "  Ruritania  " -> "Ruritania" (unknown, but whitespace-cleaned)
    assert normalize_team("  Ruritania  ") == "Ruritania"


def test_when_normalize_team_called_with_multi_space_unknown_then_spaces_are_collapsed() -> None:
    # "New   Country" -> "New Country" (unknown, inner spaces collapsed)
    assert normalize_team("New   Country") == "New Country"


# ---------------------------------------------------------------------------
# Criterion: normalize_team never raises (total function over all str inputs)
# ---------------------------------------------------------------------------


@given(st.text())
@settings(max_examples=500)
def test_when_normalize_team_called_with_any_string_then_no_exception_is_raised(
    name: str,
) -> None:
    # Must not raise for any input — criterion says "never raises"
    normalize_team(name)


# ---------------------------------------------------------------------------
# Criterion: idempotency — normalising an already-canonical value is unchanged
# (canonical names must map to themselves, so a second call is a no-op)
# ---------------------------------------------------------------------------


@given(st.text())
@settings(max_examples=500)
def test_when_normalize_team_applied_twice_then_result_equals_one_application(
    name: str,
) -> None:
    once = normalize_team(name)
    twice = normalize_team(once)
    assert once == twice


# ---------------------------------------------------------------------------
# Criterion: normalize_series preserves index and length
# ---------------------------------------------------------------------------


def test_when_normalize_series_called_then_length_is_preserved() -> None:
    s = pd.Series(["Türkiye", "USA", "UnknownNation", "Korea Republic"])
    result = normalize_series(s)
    assert len(result) == len(s)


def test_when_normalize_series_called_then_index_is_preserved() -> None:
    idx = [10, 20, 30, 40]
    s = pd.Series(["Türkiye", "USA", "UnknownNation", "Korea Republic"], index=idx)
    result = normalize_series(s)
    assert list(result.index) == idx


def test_when_normalize_series_called_with_known_aliases_then_canonical_names_are_returned() -> (
    None
):
    s = pd.Series(["Türkiye", "USA", "IR Iran", "China PR"])
    result = normalize_series(s)
    assert list(result) == ["Turkey", "United States", "Iran", "China"]


def test_when_normalize_series_called_with_unknown_names_then_cleaned_inputs_are_returned() -> None:
    s = pd.Series(["Ruritania", "Elbonia"])
    result = normalize_series(s)
    assert list(result) == ["Ruritania", "Elbonia"]


def test_when_normalize_series_called_with_mixed_series_then_each_entry_is_independently_mapped() -> (
    None
):
    s = pd.Series(["Türkiye", "Ruritania", "Cape Verde"])
    result = normalize_series(s)
    assert result[0] == "Turkey"
    assert result[1] == "Ruritania"
    assert result[2] == "Cabo Verde"


# ---------------------------------------------------------------------------
# Criterion: is_known reports whether a name maps to a canonical form
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alias, _", EXPLICIT_ALIAS_TABLE)
def test_when_is_known_called_with_known_alias_then_true_is_returned(alias: str, _: str) -> None:
    assert is_known(alias) is True


def test_when_is_known_called_with_unknown_name_then_false_is_returned() -> None:
    assert is_known("Ruritania") is False


def test_when_is_known_called_with_empty_string_then_false_is_returned() -> None:
    assert is_known("") is False


# ---------------------------------------------------------------------------
# Criterion: lookup is case-insensitive
# ---------------------------------------------------------------------------


def test_when_normalize_team_called_with_uppercase_alias_then_canonical_is_returned() -> None:
    # "TÜRKIYE" (or "turkiye") must resolve to "Turkey"
    assert normalize_team("TURKIYE") == "Turkey" or normalize_team("türkiye") == "Turkey"


def test_when_normalize_team_called_with_lowercase_usa_then_canonical_is_returned() -> None:
    assert normalize_team("usa") == "United States"


def test_when_normalize_team_called_with_mixed_case_alias_then_canonical_is_returned() -> None:
    assert normalize_team("korea REPUBLIC") == "South Korea"


def test_when_is_known_called_with_uppercase_alias_then_true_is_returned() -> None:
    assert is_known("USA") is True
    assert is_known("usa") is True
    assert is_known("Usa") is True


# ---------------------------------------------------------------------------
# Criterion: lookup is whitespace-collapsed
# ---------------------------------------------------------------------------


def test_when_normalize_team_called_with_leading_trailing_whitespace_then_canonical_is_returned() -> (
    None
):
    assert normalize_team("  USA  ") == "United States"


def test_when_normalize_team_called_with_internal_extra_whitespace_then_canonical_is_returned() -> (
    None
):
    # "Korea  Republic" (double space) should still map to "South Korea"
    assert normalize_team("Korea  Republic") == "South Korea"


def test_when_is_known_called_with_padded_alias_then_true_is_returned() -> None:
    assert is_known("  USA  ") is True


def test_when_is_known_called_with_double_spaced_alias_then_true_is_returned() -> None:
    assert is_known("Korea  Republic") is True


# ---------------------------------------------------------------------------
# Criterion: canonical names are self-consistent (canonical form maps to itself)
# This is implied by "maps … to one canonical form" — the target must be stable.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alias, canonical", EXPLICIT_ALIAS_TABLE)
def test_when_normalize_team_called_with_canonical_form_then_same_value_is_returned(
    alias: str, canonical: str
) -> None:
    assert normalize_team(canonical) == canonical
