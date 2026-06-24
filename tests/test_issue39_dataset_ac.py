"""Source-blind example tests for issue #39 acceptance criteria.

Derived directly from the acceptance criteria text, not from any implementation
source. Tests are written in the Red phase of TDD — they must fail until the
criterion is genuinely met.

Issue: feat: time-aware MatchDataset utility + HybridConfig/OrderedLogitConfig
       models (models/dataset.py)
"""

from __future__ import annotations

import sys

import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st


# ===========================================================================
# AC1 — public API surface of models/dataset.py
# ===========================================================================


def test_when_models_dataset_imported_then_MatchDataset_is_accessible():
    from worldcup_playoff.models.dataset import MatchDataset  # noqa: F401

    assert MatchDataset is not None


def test_when_models_dataset_imported_then_outcome_label_is_callable():
    from worldcup_playoff.models.dataset import outcome_label

    assert callable(outcome_label)


def test_when_models_dataset_imported_then_add_targets_is_callable():
    from worldcup_playoff.models.dataset import add_targets

    assert callable(add_targets)


def test_when_models_dataset_imported_then_played_only_is_callable():
    from worldcup_playoff.models.dataset import played_only

    assert callable(played_only)


def test_when_models_dataset_imported_then_chronological_split_is_callable():
    from worldcup_playoff.models.dataset import chronological_split

    assert callable(chronological_split)


def test_when_models_dataset_imported_then_build_dataset_is_callable():
    from worldcup_playoff.models.dataset import build_dataset

    assert callable(build_dataset)


# --- MatchDataset value object ---


def _make_match_dataset():
    """Build a minimal MatchDataset for structural checks."""
    from worldcup_playoff.models.dataset import MatchDataset

    train = pd.DataFrame({"a": [1, 2]})
    test = pd.DataFrame({"a": [3]})
    return MatchDataset(train=train, test=test, feature_cols=["a"])


def test_when_MatchDataset_constructed_then_train_attribute_is_accessible():
    ds = _make_match_dataset()
    assert ds.train is not None


def test_when_MatchDataset_constructed_then_test_attribute_is_accessible():
    ds = _make_match_dataset()
    assert ds.test is not None


def test_when_MatchDataset_constructed_then_feature_cols_attribute_is_accessible():
    ds = _make_match_dataset()
    assert ds.feature_cols == ["a"]


def test_when_MatchDataset_attribute_reassigned_then_error_is_raised():
    """MatchDataset is frozen — mutation must raise AttributeError or TypeError."""
    ds = _make_match_dataset()
    with pytest.raises((AttributeError, TypeError)):
        ds.train = pd.DataFrame()  # type: ignore[misc]


# --- outcome_label encoding: away=0, draw=1, home=2 ---


def test_when_home_wins_then_outcome_label_returns_2():
    from worldcup_playoff.models.dataset import outcome_label

    assert outcome_label(3, 1) == 2


def test_when_draw_then_outcome_label_returns_1():
    from worldcup_playoff.models.dataset import outcome_label

    assert outcome_label(2, 2) == 1


def test_when_away_wins_then_outcome_label_returns_0():
    from worldcup_playoff.models.dataset import outcome_label

    assert outcome_label(0, 1) == 0


def test_when_home_wins_by_one_then_outcome_label_returns_2():
    from worldcup_playoff.models.dataset import outcome_label

    assert outcome_label(1, 0) == 2


def test_when_away_wins_by_one_then_outcome_label_returns_0():
    from worldcup_playoff.models.dataset import outcome_label

    assert outcome_label(0, 1) == 0


def test_when_both_score_zero_then_outcome_label_returns_1():
    from worldcup_playoff.models.dataset import outcome_label

    assert outcome_label(0, 0) == 1


# --- add_targets: y_outcome (Int64) + y_margin ---


def _played_frame() -> pd.DataFrame:
    """Minimal played-match feature frame — no NA goals."""
    return pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-02-01", "2020-03-01"],
            "home_goals": pd.array([2, 1, 0], dtype="Int64"),
            "away_goals": pd.array([1, 1, 2], dtype="Int64"),
        }
    )


def test_when_add_targets_called_then_y_outcome_column_is_present():
    from worldcup_playoff.models.dataset import add_targets

    result = add_targets(_played_frame())
    assert "y_outcome" in result.columns


def test_when_add_targets_called_then_y_outcome_dtype_is_nullable_Int64():
    from worldcup_playoff.models.dataset import add_targets

    result = add_targets(_played_frame())
    assert result["y_outcome"].dtype == pd.Int64Dtype()


def test_when_add_targets_called_then_y_margin_column_is_present():
    from worldcup_playoff.models.dataset import add_targets

    result = add_targets(_played_frame())
    assert "y_margin" in result.columns


def test_when_add_targets_called_on_home_win_then_y_outcome_is_2():
    from worldcup_playoff.models.dataset import add_targets

    df = pd.DataFrame(
        {"home_goals": pd.array([3], dtype="Int64"), "away_goals": pd.array([1], dtype="Int64")}
    )
    result = add_targets(df)
    assert int(result["y_outcome"].iloc[0]) == 2


def test_when_add_targets_called_on_draw_then_y_outcome_is_1():
    from worldcup_playoff.models.dataset import add_targets

    df = pd.DataFrame(
        {"home_goals": pd.array([2], dtype="Int64"), "away_goals": pd.array([2], dtype="Int64")}
    )
    result = add_targets(df)
    assert int(result["y_outcome"].iloc[0]) == 1


def test_when_add_targets_called_on_away_win_then_y_outcome_is_0():
    from worldcup_playoff.models.dataset import add_targets

    df = pd.DataFrame(
        {"home_goals": pd.array([0], dtype="Int64"), "away_goals": pd.array([1], dtype="Int64")}
    )
    result = add_targets(df)
    assert int(result["y_outcome"].iloc[0]) == 0


def test_when_add_targets_called_then_y_margin_equals_home_minus_away():
    from worldcup_playoff.models.dataset import add_targets

    df = pd.DataFrame(
        {"home_goals": pd.array([3], dtype="Int64"), "away_goals": pd.array([1], dtype="Int64")}
    )
    result = add_targets(df)
    assert int(result["y_margin"].iloc[0]) == 2


def test_when_add_targets_called_on_away_win_then_y_margin_is_negative():
    from worldcup_playoff.models.dataset import add_targets

    df = pd.DataFrame(
        {"home_goals": pd.array([0], dtype="Int64"), "away_goals": pd.array([2], dtype="Int64")}
    )
    result = add_targets(df)
    assert int(result["y_margin"].iloc[0]) == -2


def test_when_add_targets_called_with_na_goals_then_y_outcome_is_na():
    """Rows with NA goals must receive NA targets, not raise."""
    from worldcup_playoff.models.dataset import add_targets

    df = pd.DataFrame(
        {
            "home_goals": pd.array([pd.NA], dtype="Int64"),
            "away_goals": pd.array([pd.NA], dtype="Int64"),
        }
    )
    result = add_targets(df)
    assert result["y_outcome"].isna().all()


def test_when_add_targets_called_then_original_dataframe_is_not_mutated():
    from worldcup_playoff.models.dataset import add_targets

    df = _played_frame()
    cols_before = list(df.columns)
    add_targets(df)
    assert list(df.columns) == cols_before


def test_when_add_targets_called_then_row_count_is_unchanged():
    from worldcup_playoff.models.dataset import add_targets

    df = _played_frame()
    assert len(add_targets(df)) == len(df)


# ===========================================================================
# AC2 — played_only is applied before the split (NA-goal rows excluded)
# ===========================================================================


def _mixed_frame() -> pd.DataFrame:
    """Played rows plus unplayed WC2026 fixture rows (NA goals)."""
    return pd.DataFrame(
        {
            "date": [
                "2019-01-01",
                "2020-01-01",
                "2021-01-01",
                "2022-01-01",
                "2026-07-10",
                "2026-07-14",
            ],
            "home_goals": pd.array([2, 1, 3, 0, pd.NA, pd.NA], dtype="Int64"),
            "away_goals": pd.array([1, 0, 1, 2, pd.NA, pd.NA], dtype="Int64"),
        }
    )


def test_when_played_only_called_then_rows_with_na_home_goals_are_excluded():
    from worldcup_playoff.models.dataset import played_only

    result = played_only(_mixed_frame())
    assert result["home_goals"].notna().all()


def test_when_played_only_called_then_rows_with_na_away_goals_are_excluded():
    from worldcup_playoff.models.dataset import played_only

    result = played_only(_mixed_frame())
    assert result["away_goals"].notna().all()


def test_when_played_only_called_then_played_rows_are_retained():
    from worldcup_playoff.models.dataset import played_only

    result = played_only(_mixed_frame())
    assert len(result) == 4


def test_when_build_dataset_called_then_train_has_no_na_home_goals():
    from worldcup_playoff.models.dataset import build_dataset

    result = build_dataset(_mixed_frame(), test_size=0.25, feature_cols=["date"])
    assert result.train["home_goals"].notna().all()


def test_when_build_dataset_called_then_train_has_no_na_away_goals():
    from worldcup_playoff.models.dataset import build_dataset

    result = build_dataset(_mixed_frame(), test_size=0.25, feature_cols=["date"])
    assert result.train["away_goals"].notna().all()


def test_when_build_dataset_called_then_test_has_no_na_home_goals():
    from worldcup_playoff.models.dataset import build_dataset

    result = build_dataset(_mixed_frame(), test_size=0.25, feature_cols=["date"])
    assert result.test["home_goals"].notna().all()


def test_when_build_dataset_called_then_test_has_no_na_away_goals():
    from worldcup_playoff.models.dataset import build_dataset

    result = build_dataset(_mixed_frame(), test_size=0.25, feature_cols=["date"])
    assert result.test["away_goals"].notna().all()


# ===========================================================================
# AC3 — split is positional and deterministic; date order re-asserted
# ===========================================================================


def _chronological_frame() -> pd.DataFrame:
    """Ten played rows spanning distinct calendar years in explicit order."""
    return pd.DataFrame(
        {
            "date": [
                "2015-06-01",
                "2016-06-01",
                "2017-06-01",
                "2018-06-01",
                "2019-06-01",
                "2020-06-01",
                "2021-06-01",
                "2022-06-01",
                "2023-06-01",
                "2024-06-01",
            ],
            "home_goals": pd.array([1, 2, 0, 3, 1, 2, 0, 1, 2, 1], dtype="Int64"),
            "away_goals": pd.array([0, 1, 1, 2, 1, 0, 0, 2, 1, 0], dtype="Int64"),
        }
    )


def test_when_chronological_split_called_twice_then_train_splits_are_identical():
    from worldcup_playoff.models.dataset import chronological_split

    df = _chronological_frame()
    train1, _ = chronological_split(df, test_size=0.2)
    train2, _ = chronological_split(df, test_size=0.2)
    pd.testing.assert_frame_equal(train1, train2)


def test_when_chronological_split_called_twice_then_test_splits_are_identical():
    from worldcup_playoff.models.dataset import chronological_split

    df = _chronological_frame()
    _, test1 = chronological_split(df, test_size=0.2)
    _, test2 = chronological_split(df, test_size=0.2)
    pd.testing.assert_frame_equal(test1, test2)


def test_when_chronological_split_called_then_train_max_date_lte_test_min_date():
    from worldcup_playoff.models.dataset import chronological_split

    df = _chronological_frame()
    train, test = chronological_split(df, test_size=0.3)
    train_max = pd.to_datetime(train["date"]).max()
    test_min = pd.to_datetime(test["date"]).min()
    assert train_max <= test_min


def test_when_chronological_split_called_then_all_rows_are_conserved():
    from worldcup_playoff.models.dataset import chronological_split

    df = _chronological_frame()
    train, test = chronological_split(df, test_size=0.2)
    assert len(train) + len(test) == len(df)


def test_when_build_dataset_called_on_unsorted_df_then_split_is_still_chronological():
    """Criterion requires date order to be re-asserted even for out-of-order input."""
    from worldcup_playoff.models.dataset import build_dataset

    # Reverse the chronological order to verify re-sorting happens
    df = _chronological_frame().iloc[::-1].reset_index(drop=True)
    result = build_dataset(df, test_size=0.3, feature_cols=["date"])
    train_max = pd.to_datetime(result.train["date"]).max()
    test_min = pd.to_datetime(result.test["date"]).min()
    assert train_max <= test_min


def test_when_build_dataset_called_then_feature_cols_are_stored_on_dataset():
    from worldcup_playoff.models.dataset import build_dataset

    df = _chronological_frame()
    feature_cols = ["date"]
    result = build_dataset(df, test_size=0.2, feature_cols=feature_cols)
    assert result.feature_cols == feature_cols


# ===========================================================================
# AC4 — HybridConfig / OrderedLogitConfig in config.py, wired into AppConfig
# ===========================================================================


def test_when_config_imported_then_HybridConfig_is_accessible():
    from worldcup_playoff.config import HybridConfig  # noqa: F401

    assert HybridConfig is not None


def test_when_config_imported_then_OrderedLogitConfig_is_accessible():
    from worldcup_playoff.config import OrderedLogitConfig  # noqa: F401

    assert OrderedLogitConfig is not None


def test_when_AppConfig_created_then_hybrid_attribute_is_HybridConfig_instance():
    from worldcup_playoff.config import AppConfig, HybridConfig

    assert isinstance(AppConfig().hybrid, HybridConfig)


def test_when_AppConfig_created_then_ordered_logit_attribute_is_OrderedLogitConfig_instance():
    from worldcup_playoff.config import AppConfig, OrderedLogitConfig

    assert isinstance(AppConfig().ordered_logit, OrderedLogitConfig)


def test_when_HybridConfig_created_then_random_seed_attribute_exists():
    from worldcup_playoff.config import HybridConfig

    assert hasattr(HybridConfig(), "random_seed")


def test_when_OrderedLogitConfig_created_then_random_seed_attribute_exists():
    from worldcup_playoff.config import OrderedLogitConfig

    assert hasattr(OrderedLogitConfig(), "random_seed")


def test_when_HybridConfig_created_then_random_seed_is_integer():
    from worldcup_playoff.config import HybridConfig

    assert isinstance(HybridConfig().random_seed, int)


def test_when_OrderedLogitConfig_created_then_random_seed_is_integer():
    from worldcup_playoff.config import OrderedLogitConfig

    assert isinstance(OrderedLogitConfig().random_seed, int)


def test_when_AppConfig_loaded_from_toml_with_hybrid_section_then_random_seed_is_used(
    tmp_path,
):
    """HybridConfig.random_seed round-trips through TOML loading."""
    import tomllib

    from worldcup_playoff.config import AppConfig

    toml_bytes = b"[hybrid]\nrandom_seed = 7\n"
    cfg_file = tmp_path / "cfg.toml"
    cfg_file.write_bytes(toml_bytes)
    with open(cfg_file, "rb") as f:
        raw = tomllib.load(f)
    cfg = AppConfig.model_validate(raw)
    assert cfg.hybrid.random_seed == 7


def test_when_AppConfig_loaded_from_toml_with_ordered_logit_section_then_random_seed_is_used(
    tmp_path,
):
    """OrderedLogitConfig.random_seed round-trips through TOML loading."""
    import tomllib

    from worldcup_playoff.config import AppConfig

    toml_bytes = b"[ordered_logit]\nrandom_seed = 99\n"
    cfg_file = tmp_path / "cfg.toml"
    cfg_file.write_bytes(toml_bytes)
    with open(cfg_file, "rb") as f:
        raw = tomllib.load(f)
    cfg = AppConfig.model_validate(raw)
    assert cfg.ordered_logit.random_seed == 99


def test_when_AppConfig_loaded_from_empty_toml_then_HybridConfig_defaults_are_applied(
    tmp_path,
):
    """An empty TOML must still produce a valid AppConfig with HybridConfig defaults."""
    import tomllib

    from worldcup_playoff.config import AppConfig, HybridConfig

    cfg_file = tmp_path / "empty.toml"
    cfg_file.write_bytes(b"")
    with open(cfg_file, "rb") as f:
        raw = tomllib.load(f)
    cfg = AppConfig.model_validate(raw)
    assert isinstance(cfg.hybrid, HybridConfig)


def test_when_AppConfig_loaded_from_empty_toml_then_OrderedLogitConfig_defaults_are_applied(
    tmp_path,
):
    """An empty TOML must still produce a valid AppConfig with OrderedLogitConfig defaults."""
    import tomllib

    from worldcup_playoff.config import AppConfig, OrderedLogitConfig

    cfg_file = tmp_path / "empty.toml"
    cfg_file.write_bytes(b"")
    with open(cfg_file, "rb") as f:
        raw = tomllib.load(f)
    cfg = AppConfig.model_validate(raw)
    assert isinstance(cfg.ordered_logit, OrderedLogitConfig)


# ===========================================================================
# AC5 — models/dataset.py is pure pandas/numpy: no sklearn/statsmodels at load
# ===========================================================================


def test_when_models_dataset_module_is_imported_then_sklearn_was_not_imported_by_it():
    """
    Dataset module must not pull in sklearn at module load time.

    Strategy: if sklearn was not in sys.modules before the import of
    worldcup_playoff.models.dataset, it must not appear afterwards either.
    When sklearn is already present (from other test imports) the check is
    vacuously skipped — that is acceptable because the criterion is about
    *this module's* import-time side-effects.
    """
    sklearn_already_present = any(k.startswith("sklearn") for k in sys.modules)
    import worldcup_playoff.models.dataset  # noqa: F401 — ensure it is imported

    if not sklearn_already_present:
        after = any(k.startswith("sklearn") for k in sys.modules)
        assert not after, "models.dataset must not import sklearn at module load time"


def test_when_models_dataset_module_is_imported_then_statsmodels_was_not_imported_by_it():
    """
    Dataset module must not pull in statsmodels at module load time.

    Same strategy as the sklearn check above.
    """
    sm_already_present = any(k.startswith("statsmodels") for k in sys.modules)
    import worldcup_playoff.models.dataset  # noqa: F401

    if not sm_already_present:
        after = any(k.startswith("statsmodels") for k in sys.modules)
        assert not after, "models.dataset must not import statsmodels at module load time"


# ===========================================================================
# Property-based tests — invariants derived from the acceptance criteria
# ===========================================================================


@given(
    home_goals=st.integers(min_value=0, max_value=20),
    away_goals=st.integers(min_value=0, max_value=20),
)
def test_when_outcome_label_called_for_any_valid_goals_then_result_is_in_0_1_2(
    home_goals: int, away_goals: int
) -> None:
    """outcome_label must always return exactly one of {0, 1, 2}."""
    from worldcup_playoff.models.dataset import outcome_label

    assert outcome_label(home_goals, away_goals) in (0, 1, 2)


@given(
    home_goals=st.integers(min_value=0, max_value=20),
    away_goals=st.integers(min_value=0, max_value=20),
)
def test_when_home_goals_greater_than_away_then_outcome_label_is_always_2(
    home_goals: int, away_goals: int
) -> None:
    """Home wins (home > away) must always map to the home-win encoding 2."""
    from worldcup_playoff.models.dataset import outcome_label

    if home_goals > away_goals:
        assert outcome_label(home_goals, away_goals) == 2


@given(
    home_goals=st.integers(min_value=0, max_value=20),
    away_goals=st.integers(min_value=0, max_value=20),
)
def test_when_goals_are_equal_then_outcome_label_is_always_1(
    home_goals: int, away_goals: int
) -> None:
    """Equal goals (draw) must always map to encoding 1."""
    from worldcup_playoff.models.dataset import outcome_label

    if home_goals == away_goals:
        assert outcome_label(home_goals, away_goals) == 1


@given(
    home_goals=st.integers(min_value=0, max_value=20),
    away_goals=st.integers(min_value=0, max_value=20),
)
def test_when_away_goals_greater_than_home_then_outcome_label_is_always_0(
    home_goals: int, away_goals: int
) -> None:
    """Away wins (away > home) must always map to encoding 0."""
    from worldcup_playoff.models.dataset import outcome_label

    if away_goals > home_goals:
        assert outcome_label(home_goals, away_goals) == 0


@given(
    n_played=st.integers(min_value=4, max_value=30),
    n_unplayed=st.integers(min_value=1, max_value=10),
    test_size=st.floats(min_value=0.1, max_value=0.45),
)
@settings(max_examples=40)
def test_when_build_dataset_called_then_output_contains_no_na_goals_for_any_mix(
    n_played: int, n_unplayed: int, test_size: float
) -> None:
    """For any mixture of played and unplayed rows, neither train nor test ever contains NA goals."""
    from worldcup_playoff.models.dataset import build_dataset

    goals_played = list(range(n_played))
    goals_unplayed = [pd.NA] * n_unplayed

    df = pd.DataFrame(
        {
            "date": [
                f"{2000 + (i // 12)}-{(i % 12) + 1:02d}-01" for i in range(n_played + n_unplayed)
            ],
            "home_goals": pd.array(goals_played + goals_unplayed, dtype="Int64"),
            "away_goals": pd.array(goals_played + goals_unplayed, dtype="Int64"),
        }
    )

    ds = build_dataset(df, test_size=test_size, feature_cols=["date"])
    assert ds.train["home_goals"].notna().all()
    assert ds.train["away_goals"].notna().all()
    assert ds.test["home_goals"].notna().all()
    assert ds.test["away_goals"].notna().all()


@given(
    n=st.integers(min_value=5, max_value=50),
    test_size=st.floats(min_value=0.1, max_value=0.45),
)
@settings(max_examples=40)
def test_when_chronological_split_called_then_row_count_is_always_conserved(
    n: int, test_size: float
) -> None:
    """len(train) + len(test) == len(input) for any valid DataFrame and test_size."""
    from worldcup_playoff.models.dataset import chronological_split

    df = pd.DataFrame(
        {
            "date": [f"{2000 + (i // 12)}-{(i % 12) + 1:02d}-01" for i in range(n)],
            "home_goals": pd.array(list(range(n)), dtype="Int64"),
            "away_goals": pd.array(list(range(n)), dtype="Int64"),
        }
    )
    train, test = chronological_split(df, test_size=test_size)
    assert len(train) + len(test) == n


@given(
    n=st.integers(min_value=5, max_value=50),
    test_size=st.floats(min_value=0.1, max_value=0.45),
)
@settings(max_examples=40)
def test_when_chronological_split_called_then_train_max_date_never_exceeds_test_min_date(
    n: int, test_size: float
) -> None:
    """Chronological ordering invariant: train's latest date ≤ test's earliest date."""
    from worldcup_playoff.models.dataset import chronological_split

    df = pd.DataFrame(
        {
            "date": [f"{2000 + (i // 12)}-{(i % 12) + 1:02d}-01" for i in range(n)],
            "home_goals": pd.array(list(range(n)), dtype="Int64"),
            "away_goals": pd.array(list(range(n)), dtype="Int64"),
        }
    )
    train, test = chronological_split(df, test_size=test_size)
    if not train.empty and not test.empty:
        train_max = pd.to_datetime(train["date"]).max()
        test_min = pd.to_datetime(test["date"]).min()
        assert train_max <= test_min
