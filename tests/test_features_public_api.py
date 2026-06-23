"""
Source-blind example tests for ``worldcup_playoff.features`` public API.

Issue #11 — Features package: public API, exports, and no-key integration smoke test.

All structural assertions are derived from the acceptance criteria.

Corrections from the Red-phase draft (written before implementation source was read):
- "the historical builder" (AC3) is ``build_features`` (not ``build_historical_features``).
- ``compute_elo`` and ``fit_dixon_coles`` are NOT part of ``worldcup_playoff.features``
  (they live in ``worldcup_playoff.data`` and ``worldcup_playoff.simulation``); the
  explicitly-cited-by-AC symbol check covers symbols that ARE in the features package.
- Fixture uses the uppercase internal schema (DATE/HOME_TEAM/HOME_GOALS/…) used by all
  FeatureBuilder-facing code; the raw martj42 lowercase format is handled by
  Martj42Loader.load_results() before reaching this layer.
- ``wc2026_features`` now accepts ``EloResult | pd.DataFrame`` so passing the result of
  ``compute_elo`` directly works.
"""

from __future__ import annotations

import pandas as pd
import pytest
from hypothesis import given, strategies as st

import worldcup_playoff.features as _features_pkg

# Cache __all__ at module load so @given decorators can reference it.
_ALL: list[str] = list(getattr(_features_pkg, "__all__", []))


# ---------------------------------------------------------------------------
# Criterion 1 — every name in __all__ is importable
# ---------------------------------------------------------------------------


def test_when_all_names_from_features_are_imported_then_each_name_resolves() -> None:
    """All names declared in __all__ must be accessible attributes of the package."""
    missing = [n for n in _features_pkg.__all__ if not hasattr(_features_pkg, n)]
    assert not missing, f"Names in __all__ but not importable from the package: {missing}"


def test_when_explicitly_named_ac_symbols_are_checked_then_they_appear_in_all() -> None:
    """
    Names cited in the acceptance criteria as part of the features surface must be
    in ``__all__``.

    AC: 'From build: build_features'; 'From wc2026: wc2026_features, live_fixtures_to_df';
    'From confederation: confederation_of'.
    """
    required = {"build_features", "wc2026_features", "live_fixtures_to_df", "confederation_of"}
    in_all = set(_features_pkg.__all__)
    missing = required - in_all
    assert not missing, (
        f"AC-cited symbols missing from __all__: {missing}. Current __all__: {sorted(in_all)}"
    )


@pytest.mark.skipif(not _ALL, reason="__all__ is empty — nothing to sample")
@given(st.sampled_from(_ALL or ["_sentinel"]))
def test_when_any_name_in_all_is_accessed_then_it_resolves(name: str) -> None:
    """Property: every individual name in __all__ is a real attribute on the package."""
    assert hasattr(_features_pkg, name), (
        f"{name!r} declared in __all__ but not found as a package attribute"
    )


# ---------------------------------------------------------------------------
# Criterion 2 — __all__ is sorted; no private helpers leak
# ---------------------------------------------------------------------------


def test_when_all_is_examined_then_it_is_non_empty() -> None:
    """The features package must expose at least one public symbol."""
    assert len(_features_pkg.__all__) > 0, "__all__ is empty; no public surface exposed"


def test_when_all_is_examined_then_it_is_sorted_alphabetically() -> None:
    """__all__ must equal its own sorted form (ascending lexicographic order)."""
    declared = list(_features_pkg.__all__)
    assert declared == sorted(declared), (
        f"__all__ is not sorted.\n  Expected: {sorted(declared)}\n  Got:      {declared}"
    )


@pytest.mark.skipif(not _ALL, reason="__all__ is empty — nothing to sample")
@given(st.sampled_from(_ALL or ["_sentinel"]))
def test_when_any_name_in_all_is_examined_then_no_private_helper_leaks(name: str) -> None:
    """Property: no name beginning with '_' should appear in __all__."""
    assert not name.startswith("_"), (
        f"Internal/private name {name!r} must not be exposed in __all__"
    )


# ---------------------------------------------------------------------------
# Criterion 3 — no-key integration smoke test
# ---------------------------------------------------------------------------


def _make_history() -> pd.DataFrame:
    """
    Small in-process fixture in the uppercase internal schema used by all
    FeatureBuilder-facing code.  Two played WC rows + two unplayed WC2026 rows.
    No network calls are made.
    """
    return pd.DataFrame(
        {
            "DATE": pd.to_datetime(["2022-11-20", "2022-11-24", "2026-06-11", "2026-06-15"]),
            "HOME_TEAM": ["Brazil", "France", "Brazil", "Germany"],
            "AWAY_TEAM": ["Serbia", "Australia", "France", "Spain"],
            "HOME_GOALS": pd.array([2, 4, pd.NA, pd.NA], dtype="Int64"),
            "AWAY_GOALS": pd.array([0, 1, pd.NA, pd.NA], dtype="Int64"),
            "TOURNAMENT": ["FIFA World Cup"] * 4,
            "NEUTRAL": [True, True, True, True],
        }
    )


@pytest.fixture(name="smoke_frames")
def _smoke_frames_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build (historical_df, wc2026_df) from the in-process fixture.

    compute_elo / fit_dixon_coles come from their home packages (data/simulation).
    build_features and wc2026_features are the features-package high-level API.
    Both accept EloResult directly.
    """
    from worldcup_playoff.data.elo import compute_elo
    from worldcup_playoff.features import build_features, wc2026_features
    from worldcup_playoff.simulation.poisson import fit_dixon_coles

    history = _make_history()
    elo_result = compute_elo(history)
    abilities = fit_dixon_coles(history)
    hist_df = build_features(history, elo_result, abilities)
    wc_df = wc2026_features(history, elo_result, abilities)
    return hist_df, wc_df


def test_when_no_key_smoke_test_run_then_frames_share_same_column_set(
    smoke_frames: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Historical and WC2026 feature frames must have identical column sets."""
    hist, wc = smoke_frames
    hist_cols = set(hist.columns)
    wc_cols = set(wc.columns)
    assert hist_cols == wc_cols, (
        "Column mismatch between historical and WC2026 feature frames.\n"
        f"  Only in historical: {hist_cols - wc_cols}\n"
        f"  Only in wc2026:     {wc_cols - hist_cols}"
    )


def test_when_no_key_smoke_test_run_then_wc2026_rows_are_subset_of_historical_rows(
    smoke_frames: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """
    WC2026 frame must have strictly fewer rows than the historical frame.

    The fixture has 2 played + 2 unplayed rows → build_features outputs 4 rows
    (all matches) while wc2026_features returns only the 2 unplayed rows.
    """
    hist, wc = smoke_frames
    assert len(wc) < len(hist), (
        f"WC2026 frame ({len(wc)} rows) must have strictly fewer rows than "
        f"historical frame ({len(hist)} rows)."
    )
    wc_idx = frozenset(wc.index)
    hist_idx = frozenset(hist.index)
    assert wc_idx.issubset(hist_idx), (
        "WC2026 frame index is not a subset of historical frame index.\n"
        f"  Missing from historical: {wc_idx - hist_idx}"
    )
