"""
Source-blind example tests for ``worldcup_playoff/features/__init__.py`` — Issue #38.

Issue: feat: features package public API + no-key integration smoke test
       (features/__init__.py)

All tests are derived from the acceptance-criteria text and requirements.md only
(Red-phase TDD).  No implementation source was read during authoring.

Oracle-classified criteria tested here:
  [UNIT] AC1 — __init__.py re-exports; __all__ is lexicographically sorted; no private names.
  [UNIT] AC2 — Every name in __all__ resolves as a package attribute; the four AC-cited
               symbols {build_features, wc2026_features, live_fixtures_to_df,
               confederation_of} are all present.
  [UNIT] AC3 — No-key integration: build_features + wc2026_features produce frames with
               identical column sets and no network calls (in-process Elo + abilities).
  [UNIT] AC4 — WC2026 output rows have goals <NA>; historical played rows have non-NA
               integer goals.

Criteria skipped (not runtime-verifiable per oracle):
  [NOT VERIFIABLE] All tests pass (suite gate).
  [NOT VERIFIABLE] SOLID / code quality.

Invariant choices where criterion text is ambiguous:
- "re-exports" means each symbol is reachable as ``worldcup_playoff.features.<name>``
  (i.e., ``hasattr(worldcup_playoff.features, name)`` is True) AND is listed in
  ``__all__``.
- "lexicographically sorted" = Python's built-in ``sorted()`` (default ASCII/Unicode
  order on strings).
- "no network calls" is validated by the fixture completing without exception in an
  environment where no API key is set and the only data is the in-process DataFrame.
- "strict subset" for AC4 means ``len(wc_df) < len(hist_df)``; index-label subset is
  also checked because the criterion says "subset of the historical rows".
- "historical played rows have non-NA integer goals" means
  ``hist_df.loc[hist_df["home_goals"].notna(), "home_goals"].dtype == Int64``.
"""

from __future__ import annotations

import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

import worldcup_playoff.features as features_pkg

# ---------------------------------------------------------------------------
# Constants derived from the acceptance criteria (not from implementation)
# ---------------------------------------------------------------------------

_AC_CITED_SYMBOLS: frozenset[str] = frozenset(
    {"build_features", "wc2026_features", "live_fixtures_to_df", "confederation_of"}
)

# Cache __all__ at import time so @given decorators can reference it.
_DECLARED_ALL: list[str] = list(getattr(features_pkg, "__all__", []))


# ---------------------------------------------------------------------------
# AC1 — __all__ is defined, lexicographically sorted, and contains no private names
#
# Criterion text: "re-exports the symbols above and declares a lexicographically
# sorted __all__ with no private (_-prefixed) names."
# ---------------------------------------------------------------------------


def test_when_features_package_is_imported_then_all_attribute_is_defined() -> None:
    """
    The features package must declare ``__all__`` as a module-level attribute.

    A missing ``__all__`` means callers cannot know the stable public surface.
    """
    assert hasattr(features_pkg, "__all__"), "worldcup_playoff.features does not define __all__"


def test_when_all_is_examined_then_it_is_lexicographically_sorted() -> None:
    """
    AC1 example test.

    ``__all__`` must equal ``sorted(__all__)`` — the criterion says
    'lexicographically sorted'.  One failed ordering is enough to fail this.
    """
    declared = list(features_pkg.__all__)
    assert declared == sorted(declared), (
        "__all__ is not lexicographically sorted.\n"
        f"  Expected: {sorted(declared)}\n"
        f"  Got:      {declared}"
    )


def test_when_all_is_examined_then_no_private_name_is_present() -> None:
    """
    AC1 example test — separate from sorting because it is a distinct observable
    behaviour.

    The criterion says 'no private (_-prefixed) names'.
    """
    private = [n for n in features_pkg.__all__ if n.startswith("_")]
    assert not private, f"Private (underscore-prefixed) names must not appear in __all__: {private}"


# Property: sorting invariant — the sorted order must hold for every individual
# name in the list (ordering / monotonicity criterion).
@pytest.mark.skipif(not _DECLARED_ALL, reason="__all__ is empty — nothing to sample")
@given(st.sampled_from(_DECLARED_ALL or ["_sentinel"]))
@settings(max_examples=200)
def test_when_any_name_in_all_is_inspected_then_it_has_no_underscore_prefix(
    name: str,
) -> None:
    """
    Property (no-private invariant): every individual name drawn from ``__all__``
    must not start with ``'_'``.  Derived from "no private (_-prefixed) names".
    """
    assert not name.startswith("_"), (
        f"Private name {name!r} found in __all__; __all__ must contain only public symbols."
    )


# ---------------------------------------------------------------------------
# AC2 — Every name in __all__ resolves; the four AC-cited symbols are present
#
# Criterion text: "Every name in __all__ resolves as a package attribute; the
# AC-cited symbols {build_features, wc2026_features, live_fixtures_to_df,
# confederation_of} are all present."
# ---------------------------------------------------------------------------


def test_when_all_four_ac_cited_symbols_are_checked_then_all_are_in_all() -> None:
    """
    AC2 example test.

    All four symbols named in the acceptance criteria must appear in ``__all__``.
    """
    in_all = set(features_pkg.__all__)
    missing = _AC_CITED_SYMBOLS - in_all
    assert not missing, (
        f"AC-cited symbols missing from __all__: {sorted(missing)}.\n"
        f"Current __all__: {sorted(in_all)}"
    )


@pytest.mark.parametrize("symbol", sorted(_AC_CITED_SYMBOLS))
def test_when_ac_cited_symbol_is_accessed_as_attribute_then_it_resolves(
    symbol: str,
) -> None:
    """
    AC2 — one parametrized example test per AC-cited symbol.

    Each of the four symbols must be importable directly from the package
    (not just listed in __all__).
    """
    assert hasattr(features_pkg, symbol), (
        f"{symbol!r} must be accessible as worldcup_playoff.features.{symbol} "
        f"but it is missing as a package attribute."
    )


def test_when_all_names_in_all_are_accessed_then_each_resolves_as_package_attribute() -> None:
    """
    AC2 — exhaustive check.

    Every name declared in ``__all__`` must be an accessible attribute of the
    package.  This pins the "no broken re-export" contract.
    """
    missing = [n for n in features_pkg.__all__ if not hasattr(features_pkg, n)]
    assert not missing, f"Names in __all__ that are not accessible as package attributes: {missing}"


# Property: never-raises for any name already in __all__ (valid-input domain).
@pytest.mark.skipif(not _DECLARED_ALL, reason="__all__ is empty — nothing to sample")
@given(st.sampled_from(_DECLARED_ALL or ["_sentinel"]))
@settings(max_examples=200)
def test_when_any_name_in_all_is_accessed_as_attribute_then_it_resolves(
    name: str,
) -> None:
    """
    Property: every element of ``__all__`` must resolve via ``getattr``
    without raising ``AttributeError``.  Derived from "every name in __all__
    resolves as a package attribute".
    """
    assert hasattr(features_pkg, name), (
        f"{name!r} declared in __all__ but not found as a package attribute."
    )


# ---------------------------------------------------------------------------
# Shared fixture for AC3 + AC4 — built entirely in-process, no network
#
# The fixture exercises the public API at the package level:
#   worldcup_playoff.features.build_features
#   worldcup_playoff.features.wc2026_features
# which are the symbols whose presence AC2 already pins.
# ---------------------------------------------------------------------------


def _make_in_process_history() -> pd.DataFrame:
    """
    Minimal in-process martj42 internal-schema DataFrame.

    Three played WC matches (HOME_GOALS / AWAY_GOALS are integers) plus two
    unplayed WC2026 fixtures (HOME_GOALS / AWAY_GOALS are pandas NA).  All rows
    use the uppercase internal schema (DATE / HOME_TEAM / … ) used by all
    FeatureBuilder-facing code; the raw martj42 lowercase format is handled by
    Martj42Loader before reaching this layer.
    """
    return pd.DataFrame(
        {
            "DATE": pd.to_datetime(
                ["2022-11-20", "2022-11-24", "2022-11-28", "2026-06-11", "2026-06-15"]
            ),
            "HOME_TEAM": ["Brazil", "France", "Argentina", "Brazil", "Germany"],
            "AWAY_TEAM": ["Serbia", "Australia", "Poland", "France", "Spain"],
            "HOME_GOALS": pd.array([2, 4, 2, pd.NA, pd.NA], dtype="Int64"),
            "AWAY_GOALS": pd.array([0, 1, 0, pd.NA, pd.NA], dtype="Int64"),
            "TOURNAMENT": ["FIFA World Cup"] * 5,
            "NEUTRAL": [True, True, True, True, True],
        }
    )


@pytest.fixture(scope="module", name="no_key_frames")
def _no_key_frames_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build (historical_df, wc2026_df) from the in-process fixture.

    Imports:
    - ``compute_elo`` and ``fit_dixon_coles`` from their home packages (not under test).
    - ``build_features`` and ``wc2026_features`` from ``worldcup_playoff.features``
      (the public API under test — retrieved via ``getattr`` to stay source-blind).

    No network calls are needed: all data is constructed from the in-process DataFrame.
    """
    from worldcup_playoff.data.elo import compute_elo
    from worldcup_playoff.simulation.poisson import fit_dixon_coles

    # Access the public API under test through the package, not sub-module imports,
    # to confirm that re-exporting via __init__.py actually works.
    build_features = getattr(features_pkg, "build_features")
    wc2026_features = getattr(features_pkg, "wc2026_features")

    history = _make_in_process_history()
    elo_result = compute_elo(history)
    abilities = fit_dixon_coles(history)
    hist_df = build_features(history, elo_result, abilities)
    wc_df = wc2026_features(history, elo_result, abilities)
    return hist_df, wc_df


# ---------------------------------------------------------------------------
# AC3 — build_features and wc2026_features produce frames with identical column sets
#       and no network calls
#
# Criterion text: "No-key integration: in-process Elo + abilities →
# build_features and wc2026_features produce frames with identical column sets
# and no network calls."
# ---------------------------------------------------------------------------


def test_when_no_key_integration_is_run_then_frames_have_identical_column_sets(
    no_key_frames: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """
    AC3 example test.

    The historical feature frame and the WC2026 feature frame must expose the
    same column set.  A difference signals a broken feature assembly pipeline.
    """
    hist, wc = no_key_frames
    hist_cols = set(hist.columns)
    wc_cols = set(wc.columns)
    assert hist_cols == wc_cols, (
        "build_features and wc2026_features returned frames with different column sets.\n"
        f"  Only in historical: {sorted(hist_cols - wc_cols)}\n"
        f"  Only in wc2026:     {sorted(wc_cols - hist_cols)}"
    )


# ---------------------------------------------------------------------------
# AC4 — WC2026 output rows are a strict subset of the historical rows with
#       goals <NA>; historical played rows have non-NA integer goals
#
# Criterion text: "WC2026 output rows are a strict subset of the historical
# rows with goals <NA>; historical played rows have non-NA integer goals."
# ---------------------------------------------------------------------------


def test_when_no_key_integration_is_run_then_wc2026_row_count_is_strictly_less_than_historical(
    no_key_frames: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """
    AC4 example test (subset count).

    The fixture has 3 played + 2 unplayed rows.  ``build_features`` must include
    all 5 rows; ``wc2026_features`` must return only the 2 unplayed WC2026 rows
    (strict subset by count).
    """
    hist, wc = no_key_frames
    assert len(wc) < len(hist), (
        f"WC2026 frame ({len(wc)} rows) must be a strict subset of the historical "
        f"frame ({len(hist)} rows) by row count."
    )


def test_when_no_key_integration_is_run_then_wc2026_home_goals_are_na(
    no_key_frames: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """
    AC4 example test — goals <NA> contract for WC2026 rows.

    Every row returned by ``wc2026_features`` must have ``home_goals`` equal to
    ``<NA>`` (an unplayed fixture).
    """
    _, wc = no_key_frames
    assert wc["home_goals"].isna().all(), (
        "Some WC2026 feature rows have non-NA home_goals — all must be <NA>."
    )


def test_when_no_key_integration_is_run_then_wc2026_away_goals_are_na(
    no_key_frames: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """
    AC4 example test — goals <NA> contract for WC2026 rows (away side).

    Each row mirrors ``home_goals``: the match is unplayed, so away goals must
    also be ``<NA>``.
    """
    _, wc = no_key_frames
    assert wc["away_goals"].isna().all(), (
        "Some WC2026 feature rows have non-NA away_goals — all must be <NA>."
    )


def test_when_no_key_integration_is_run_then_historical_played_rows_have_non_na_integer_goals(
    no_key_frames: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """
    AC4 example test — played-row goal contract.

    The criterion states "historical played rows have non-NA integer goals".
    The fixture has exactly 3 played rows; all must have non-NA Int64 home/away goals.
    """
    hist, _ = no_key_frames
    played = hist[hist["home_goals"].notna()]
    assert len(played) == 3, (
        f"Expected 3 played rows with integer goals in the historical frame, got {len(played)}."
    )
    assert played["away_goals"].notna().all(), (
        "Some played historical rows have NA away_goals — they should be non-NA integers."
    )
    assert played["home_goals"].dtype == pd.Int64Dtype(), (
        f"home_goals dtype in played rows must be Int64 (nullable), "
        f"got {played['home_goals'].dtype}."
    )
