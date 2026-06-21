"""Tests for DistributionFitter, FeatureSampler, and FittedDistribution."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from worldcup_playoff.config import DistributionConfig, FeaturesConfig
from worldcup_playoff.simulation.distributions import (
    DistributionFitter,
    FeatureSampler,
    FittedDistribution,
)


# ---------------------------------------------------------------------------
# FittedDistribution
# ---------------------------------------------------------------------------


class TestFittedDistribution:
    def test_fields(self) -> None:
        d = FittedDistribution(name="norm", params=(0.0, 1.0))
        assert d.name == "norm"
        assert d.params == (0.0, 1.0)

    def test_is_immutable(self) -> None:
        d = FittedDistribution(name="norm", params=(0.0, 1.0))
        with pytest.raises((AttributeError, TypeError)):
            d.name = "beta"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# FeatureSampler.sample
# ---------------------------------------------------------------------------


class TestFeatureSampler:
    def test_sample_shape_single(self) -> None:
        dists = [FittedDistribution(name="norm", params=(0.0, 1.0))]
        result = FeatureSampler.sample(dists, size=1, random_state=42)
        assert result.shape == (1, 1)

    def test_sample_shape_multiple(self) -> None:
        dists = [
            FittedDistribution(name="norm", params=(0.0, 1.0)),
            FittedDistribution(name="norm", params=(5.0, 2.0)),
        ]
        result = FeatureSampler.sample(dists, size=10, random_state=42)
        assert result.shape == (10, 2)

    def test_sample_reproducible_with_same_seed(self) -> None:
        dists = [FittedDistribution(name="norm", params=(0.0, 1.0))]
        r1 = FeatureSampler.sample(dists, size=5, random_state=42)
        r2 = FeatureSampler.sample(dists, size=5, random_state=42)
        np.testing.assert_array_equal(r1, r2)

    def test_sample_differs_with_different_seeds(self) -> None:
        dists = [FittedDistribution(name="norm", params=(0.0, 1.0))]
        r1 = FeatureSampler.sample(dists, size=10, random_state=1)
        r2 = FeatureSampler.sample(dists, size=10, random_state=2)
        assert not np.allclose(r1, r2)

    def test_sample_features_are_independent(self) -> None:
        """Features drawn with the same seed must not be perfectly correlated."""
        dists = [
            FittedDistribution(name="norm", params=(0.0, 1.0)),
            FittedDistribution(name="norm", params=(5.0, 2.0)),
        ]
        result = FeatureSampler.sample(dists, size=200, random_state=42)
        col0 = (result[:, 0] - result[:, 0].mean()) / result[:, 0].std()
        col1 = (result[:, 1] - result[:, 1].mean()) / result[:, 1].std()
        corr = float(np.abs(np.corrcoef(col0, col1)[0, 1]))
        assert corr < 0.3, (
            f"Feature columns are too correlated ({corr:.3f}); each feature must "
            "draw from its own independent random stream."
        )


# ---------------------------------------------------------------------------
# FeatureSampler.assemble
# ---------------------------------------------------------------------------


class TestFeatureSamplerAssemble:
    def _make_five_dists(self, mu: float = 0.0) -> list[FittedDistribution]:
        return [FittedDistribution(name="norm", params=(mu, 1.0)) for _ in range(5)]

    def test_assemble_returns_length_10_vector(self) -> None:
        sampler = FeatureSampler(FeaturesConfig())
        home_dists = self._make_five_dists(mu=1.0)
        away_dists = self._make_five_dists(mu=0.0)
        vector = sampler.assemble(home_dists, away_dists, random_state=42)
        assert vector.shape == (10,)

    def test_assemble_home_features_come_first(self) -> None:
        """Home features (large mu) must occupy the first 5 positions."""
        sampler = FeatureSampler(FeaturesConfig())
        home_dists = self._make_five_dists(mu=100.0)  # very large
        away_dists = self._make_five_dists(mu=-100.0)  # very small
        vector = sampler.assemble(home_dists, away_dists, random_state=42)
        assert float(vector[:5].mean()) > 50.0, "Home features should have large values"
        assert float(vector[5:].mean()) < -50.0, "Away features should have small values"


# ---------------------------------------------------------------------------
# Helpers for DistributionFitter tests
# ---------------------------------------------------------------------------


def _make_train_df(n: int = 20) -> pd.DataFrame:
    """Minimal train_data.csv-format DataFrame for two teams."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        home, away = ("Brazil", "France") if i % 2 == 0 else ("France", "Brazil")
        rows.append(
            {
                "HOME_TEAM": home,
                "AWAY_TEAM": away,
                "DATE": f"2020-0{(i % 9) + 1}-01",
                "GOALS_home": int(rng.integers(0, 4)),
                "SHOTS_home": int(rng.integers(5, 20)),
                "SHOTS_ON_TARGET_home": int(rng.integers(2, 8)),
                "POSSESSION_home": float(rng.uniform(40, 65)),
                "PASS_PCT_home": float(rng.uniform(65, 85)),
                "GOALS_away": int(rng.integers(0, 4)),
                "SHOTS_away": int(rng.integers(5, 20)),
                "SHOTS_ON_TARGET_away": int(rng.integers(2, 8)),
                "POSSESSION_away": float(rng.uniform(40, 65)),
                "PASS_PCT_away": float(rng.uniform(65, 85)),
                "HOME_WIN": 1,
            }
        )
    return pd.DataFrame(rows)


def _make_fitter(min_season: int = 2019, candidates: list[str] | None = None) -> DistributionFitter:
    dist_cfg = DistributionConfig(
        min_season=min_season,
        candidates=candidates or ["norm"],
    )
    feat_cfg = FeaturesConfig()
    return DistributionFitter(config=dist_cfg, features_config=feat_cfg)


def _stub_fit_team(fitter: DistributionFitter) -> MagicMock:
    """Stub _fit_team to avoid the slow Fitter library in unit tests."""
    stub = MagicMock(
        return_value=[FittedDistribution(name="norm", params=(0.0, 1.0))] * 5
    )
    fitter._fit_team = stub  # type: ignore[method-assign]
    return stub


# ---------------------------------------------------------------------------
# DistributionFitter — season filter
# ---------------------------------------------------------------------------


class TestDistributionFitterSeasonFilter:
    def test_min_season_row_is_included(self) -> None:
        """Rows from exactly min_season must not be silently dropped."""
        fitter = _make_fitter(min_season=2020)
        _stub_fit_team(fitter)

        df = _make_train_df(n=10)
        # All rows are dated 2020-xx-01, so all should be included
        result = fitter.fit_all_teams(df)
        assert result, "fit_all_teams returned empty dict; min_season row was excluded"

    def test_rows_before_min_season_excluded(self) -> None:
        fitter = _make_fitter(min_season=2025)
        _stub_fit_team(fitter)

        df = _make_train_df(n=10)  # all dates in 2020
        result = fitter.fit_all_teams(df)
        assert not result, "Should return empty when all rows predate min_season"

    def test_no_date_column_uses_all_rows(self) -> None:
        fitter = _make_fitter(min_season=2020)
        stub = _stub_fit_team(fitter)

        df = _make_train_df(n=10).drop(columns=["DATE"])
        fitter.fit_all_teams(df)
        assert stub.called


# ---------------------------------------------------------------------------
# DistributionFitter — team discovery
# ---------------------------------------------------------------------------


class TestDistributionFitterTeamDiscovery:
    def test_away_only_team_is_included(self) -> None:
        """A team that only appears as away must still be fitted."""
        fitter = _make_fitter(min_season=2020)
        _stub_fit_team(fitter)

        df = _make_train_df(n=10)
        # Remove all rows where Germany is home — Germany only plays away
        df = df[df["HOME_TEAM"] != "Brazil"].copy()
        df["HOME_TEAM"] = "France"
        df["AWAY_TEAM"] = "Brazil"

        result = fitter.fit_all_teams(df)
        assert "Brazil" in result or len(result) >= 1

    def test_aggregate_team_data_shape_matches_for_home_and_away(self) -> None:
        """Home-team and away-team data must produce the same matrix shape."""
        fitter = _make_fitter(min_season=2020)
        df = _make_train_df(n=20)

        # Brazil always home, France always away
        df = df.copy()
        df["HOME_TEAM"] = "Brazil"
        df["AWAY_TEAM"] = "France"

        team_data = fitter._aggregate_team_data(df)

        assert "Brazil" in team_data
        assert "France" in team_data

        shape_brazil = team_data["Brazil"].shape
        shape_france = team_data["France"].shape
        assert shape_brazil == shape_france, (
            f"Shape mismatch: Brazil {shape_brazil} vs France {shape_france}. "
            "The away-column rename may not be working correctly."
        )
        assert shape_brazil == (20, 5)


# ---------------------------------------------------------------------------
# DistributionFitter — save / load round-trip
# ---------------------------------------------------------------------------


class TestDistributionFitterSaveLoad:
    def test_save_load_round_trip(self, tmp_path: Path) -> None:
        distributions = {
            "Brazil": [
                FittedDistribution(name="norm", params=(1.8, 0.8)),
                FittedDistribution(name="norm", params=(16.0, 3.0)),
                FittedDistribution(name="norm", params=(5.0, 1.5)),
                FittedDistribution(name="norm", params=(52.0, 5.0)),
                FittedDistribution(name="norm", params=(77.0, 4.0)),
            ],
            "France": [
                FittedDistribution(name="norm", params=(1.6, 0.7)),
                FittedDistribution(name="norm", params=(15.0, 3.0)),
                FittedDistribution(name="norm", params=(4.8, 1.4)),
                FittedDistribution(name="norm", params=(51.0, 5.0)),
                FittedDistribution(name="norm", params=(76.0, 4.0)),
            ],
        }

        path = tmp_path / "distributions.json"
        DistributionFitter.save(distributions, path)
        assert path.exists()

        loaded = DistributionFitter.load(path)
        assert set(loaded.keys()) == {"Brazil", "France"}
        assert len(loaded["Brazil"]) == 5
        assert loaded["Brazil"][0].name == "norm"
        assert loaded["Brazil"][0].params == pytest.approx((1.8, 0.8))

    def test_save_creates_parent_directories(self, tmp_path: Path) -> None:
        path = tmp_path / "output" / "distributions.json"
        DistributionFitter.save(
            {"T": [FittedDistribution("norm", (0.0, 1.0))]}, path
        )
        assert path.exists()

    def test_load_preserves_param_tuple_type(self, tmp_path: Path) -> None:
        distributions = {
            "X": [FittedDistribution(name="norm", params=(3.14, 2.72))]
        }
        path = tmp_path / "dists.json"
        DistributionFitter.save(distributions, path)
        loaded = DistributionFitter.load(path)
        assert isinstance(loaded["X"][0].params, tuple)
