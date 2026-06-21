"""Statistical distribution fitting and feature sampling for football match data.

Mirrors ``nba_playoff.simulation.distributions`` but adapts team-data aggregation
to the football train_data.csv schema: columns are ``HOME_TEAM``, ``AWAY_TEAM``,
10 feature columns (5 home + 5 away), and ``HOME_WIN``.  Per-team historical
observations are gathered by combining the *_home columns from rows where the
team played at home with the *_away columns from rows where the team played away.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from fitter import Fitter
from scipy import stats

from worldcup_playoff.config import DistributionConfig, FeaturesConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FittedDistribution:
    """Immutable record of a single fitted statistical distribution."""

    name: str
    params: tuple[float, ...]


# Maps distribution names to scipy random variate samplers.
# Mirrors the SCIPY_GENERATORS dict in the NBA original.
SCIPY_GENERATORS: dict[str, Callable[..., Any]] = {
    "alpha": stats.alpha.rvs,
    "beta": stats.beta.rvs,
    "chi": stats.chi.rvs,
    "cosine": stats.cosine.rvs,
    "dgamma": stats.dgamma.rvs,
    "dweibull": stats.dweibull.rvs,
    "f": stats.f.rvs,
    "fisk": stats.fisk.rvs,
    "gamma": stats.gamma.rvs,
    "maxwell": stats.maxwell.rvs,
    "norm": stats.norm.rvs,
    "pareto": stats.pareto.rvs,
    "t": stats.t.rvs,
}


class DistributionFitter:
    """Fits statistical distributions to per-team football match data.

    Unlike the NBA fitter (which pivots on ``HOME_TEAM_ID`` / ``VISITOR_TEAM_ID``
    and a ``SEASON`` column), this class reads from the football train_data.csv
    schema:

    - Column names for home perspective: ``GOALS_home``, ``SHOTS_home``, …
    - Column names for away perspective: ``GOALS_away``, ``SHOTS_away``, …
    - Team identification columns: ``HOME_TEAM``, ``AWAY_TEAM``
    - No ``SEASON`` column — ``DistributionConfig.min_season`` is interpreted as
      the minimum calendar year extracted from an optional ``DATE`` column.  When
      no ``DATE`` column exists all rows are used.

    For each team ``T`` and each of the five base statistics the fitter gathers:

    - The ``*_home`` column values from rows where ``HOME_TEAM == T``
    - The ``*_away`` column values from rows where ``AWAY_TEAM == T``

    Both slices are stacked vertically so that the fitted distribution captures
    the team's performance regardless of venue.
    """

    def __init__(
        self,
        config: DistributionConfig,
        features_config: FeaturesConfig,
    ) -> None:
        self._config = config
        self._features = features_config

    def fit_all_teams(
        self,
        df: pd.DataFrame,
    ) -> dict[str, list[FittedDistribution]]:
        """Fit distributions for each team found in *df*.

        Args:
            df: DataFrame conforming to the train_data.csv schema.  Must contain
                ``HOME_TEAM``, ``AWAY_TEAM``, and the 10 feature columns listed in
                ``FeaturesConfig.selected``.  An optional ``DATE`` column (ISO
                format) is used to filter by ``DistributionConfig.min_season``.

        Returns:
            Mapping of ``{team_name: [FittedDistribution per base stat]}``.
            The list length equals ``FeaturesConfig.per_team_count`` (5).
        """
        filtered_df = self._filter_by_season(df)
        team_data = self._aggregate_team_data(filtered_df)

        result: dict[str, list[FittedDistribution]] = {}
        for team_name, data in team_data.items():
            logger.info("Fitting distributions for %s", team_name)
            result[team_name] = self._fit_team(data)

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _filter_by_season(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return only rows at or after ``min_season``.

        Falls back to the full DataFrame when no ``DATE`` column is present.
        """
        if "DATE" not in df.columns:
            logger.debug(
                "No DATE column found; using all %d rows for distribution fitting",
                len(df),
            )
            return df.reset_index(drop=True)

        years = pd.to_datetime(df["DATE"], errors="coerce").dt.year
        mask = years >= self._config.min_season
        filtered = df.loc[mask].reset_index(drop=True)
        logger.debug(
            "Filtered to %d rows from year >= %d", len(filtered), self._config.min_season
        )
        return filtered

    def _aggregate_team_data(
        self,
        df: pd.DataFrame,
    ) -> dict[str, np.ndarray]:
        """Build a per-team observation matrix.

        For each unique team, combines:
        - Rows where the team was the home side → the five ``*_home`` columns
        - Rows where the team was the away side → the five ``*_away`` columns

        Returns ``{team_name: ndarray of shape (n_observations, per_team_count)}``.
        """
        n = self._features.per_team_count
        selected = self._features.selected
        home_features: list[str] = selected[:n]   # GOALS_home … PASS_PCT_home
        away_features: list[str] = selected[n:]   # GOALS_away … PASS_PCT_away

        unique_teams: np.ndarray = pd.unique(
            pd.concat(  # type: ignore[arg-type]
                [df["HOME_TEAM"], df["AWAY_TEAM"]]
            )
        )

        team_data: dict[str, np.ndarray] = {}
        for team in unique_teams:
            # Games played at home — take the _home columns directly.
            df_home = df.loc[df["HOME_TEAM"] == team][home_features]

            # Games played away — take the _away columns and re-label them so
            # both slices share the same column names before stacking.
            df_away = df.loc[df["AWAY_TEAM"] == team][away_features].copy()
            df_away.columns = home_features  # type: ignore[assignment]

            combined = pd.concat([df_home, df_away], axis=0)
            team_data[str(team)] = combined.to_numpy()

        return team_data

    def _fit_team(self, data: np.ndarray) -> list[FittedDistribution]:
        """Fit one distribution per base statistic for a single team.

        Args:
            data: Array of shape ``(n_observations, per_team_count)``.

        Returns:
            List of ``FittedDistribution`` with length ``per_team_count``.
        """
        distributions: list[FittedDistribution] = []
        n_features = self._features.per_team_count

        for i in range(n_features):
            f = Fitter(data[:, i])
            f.distributions = self._config.candidates
            f.fit()
            best = f.get_best(method="sumsquare_error")
            dist_name = list(best.keys())[0]
            raw_params = best[dist_name]
            dist_params = (
                tuple(raw_params.values())
                if isinstance(raw_params, dict)
                else tuple(raw_params)
            )
            distributions.append(
                FittedDistribution(
                    name=dist_name,
                    params=tuple(float(p) for p in dist_params),
                )
            )

        return distributions

    @staticmethod
    def save(
        team_distributions: dict[str, list[FittedDistribution]],
        path: Path,
    ) -> None:
        """Serialize fitted distributions to JSON.

        Args:
            team_distributions: The mapping returned by ``fit_all_teams``.
            path: Destination path; parent directories are created automatically.
        """
        serializable = {
            team: [{"name": d.name, "params": list(d.params)} for d in dists]
            for team, dists in team_distributions.items()
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(serializable, fh, indent=2)
        logger.info("Distributions saved to %s", path)

    @staticmethod
    def load(path: Path) -> dict[str, list[FittedDistribution]]:
        """Deserialize fitted distributions from JSON.

        Args:
            path: Path to a JSON file previously produced by ``save``.

        Returns:
            Mapping of ``{team_name: [FittedDistribution per base stat]}``.
        """
        with open(path) as fh:
            raw = json.load(fh)
        return {
            team: [
                FittedDistribution(name=d["name"], params=tuple(d["params"]))
                for d in dists
            ]
            for team, dists in raw.items()
        }


class FeatureSampler:
    """Assembles synthetic feature vectors from two teams' fitted distributions.

    The sampling contract mirrors the NBA ``FeatureSampler`` but is adapted for
    the football feature layout.  A World Cup feature vector is ordered exactly
    as ``FeaturesConfig.selected``:

    ``[GOALS_home, SHOTS_home, SHOTS_ON_TARGET_home, POSSESSION_home, PASS_PCT_home,``
    ``GOALS_away, SHOTS_away, SHOTS_ON_TARGET_away, POSSESSION_away, PASS_PCT_away]``

    Usage::

        sampler = FeatureSampler(features_config)
        vector = sampler.assemble(home_dists, away_dists)
    """

    def __init__(self, features_config: FeaturesConfig) -> None:
        self._features = features_config

    def assemble(
        self,
        home_distributions: list[FittedDistribution],
        away_distributions: list[FittedDistribution],
        random_state: int | None = None,
    ) -> np.ndarray:
        """Sample one feature vector for a single knockout tie.

        Draws one value per base statistic for the home team (first 5 positions)
        and one value per base statistic for the away team (last 5 positions),
        producing a 1-D array of length ``2 * per_team_count`` (i.e. 10).

        Args:
            home_distributions: Fitted distributions for the home team, ordered
                by base stat index matching ``FeaturesConfig.selected[:per_team_count]``.
            away_distributions: Fitted distributions for the away team, ordered
                by base stat index matching ``FeaturesConfig.selected[per_team_count:]``.
            random_state: Optional seed for reproducible sampling.

        Returns:
            1-D ``np.ndarray`` of shape ``(10,)`` ready for ``classifier.predict``.
        """
        home_samples = self._sample_team(home_distributions, random_state=random_state)
        away_samples = self._sample_team(away_distributions, random_state=random_state)
        return np.concatenate([home_samples, away_samples])

    @staticmethod
    def sample(
        distributions: list[FittedDistribution],
        size: int = 1,
        random_state: int | None = None,
    ) -> np.ndarray:
        """Sample ``size`` observations for each feature from *distributions*.

        This low-level helper mirrors the NBA ``FeatureSampler.sample`` signature
        exactly so that ``GamePredictor`` can share the same call site pattern.

        Args:
            distributions: List of fitted distributions, one per feature.
            size: Number of samples to draw per feature.
            random_state: Optional seed for reproducibility.

        Returns:
            Array of shape ``(size, n_features)``.
        """
        rng = np.random.default_rng(random_state)
        samples: list[Any] = []
        for dist in distributions:
            generator = SCIPY_GENERATORS[dist.name]
            s = generator(*dist.params, size=size, random_state=rng)
            samples.append(s)
        return np.array(samples).T

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sample_team(
        self,
        distributions: list[FittedDistribution],
        random_state: int | None = None,
    ) -> np.ndarray:
        """Return a 1-D array of one sampled value per base stat for one team."""
        # sample() returns shape (1, n_features); squeeze to (n_features,)
        return self.sample(distributions, size=1, random_state=random_state).squeeze(axis=0)
