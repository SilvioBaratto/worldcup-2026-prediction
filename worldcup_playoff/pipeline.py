"""Pipeline orchestrator — ties all stages together.

Mirrors ``nba_playoff.pipeline`` but adapted for FIFA World Cup 2026 single-
elimination knockout prediction:

- ``NBAClient`` -> ``FootballClient``
- ``GamesBuilder`` / ``games.csv`` -> ``MatchesBuilder`` / ``matches.csv``
- ``build_games_details_csv`` -> ``build_match_details_csv``
- best-of-7 series fields removed; ``SimulationConfig`` carries only
  ``n_simulations`` and ``classifier``.
- ``DataCleaner.clean`` accepts ``(matches_df, details_df)`` — no ``teams_df``.
- ``FeatureSampler`` requires ``FeaturesConfig`` at construction time.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from worldcup_playoff.config import AppConfig, BracketConfig
from worldcup_playoff.data.builders import (
    MatchesBuilder,
    TeamsBuilder,
    build_match_details_csv,
)
from worldcup_playoff.data.cleaner import DataCleaner
from worldcup_playoff.data.client import FootballClient
from worldcup_playoff.data.loader import DataLoader
from worldcup_playoff.models.classifiers import ClassifierFactory, ClassifierTrainer
from worldcup_playoff.models.evaluation import ModelEvaluator
from worldcup_playoff.simulation.distributions import (
    DistributionFitter,
    FeatureSampler,
    FittedDistribution,
)
from worldcup_playoff.simulation.game import GamePredictor
from worldcup_playoff.simulation.tournament import RoundResult, TournamentSimulator
from worldcup_playoff.types import Classifier

logger = logging.getLogger(__name__)


class Pipeline:
    """Orchestrates the full clean -> train -> simulate -> visualize workflow.

    Each public method is independently runnable so that individual pipeline
    stages can be re-executed without repeating expensive upstream steps.

    Args:
        config: Full application configuration loaded from TOML.
        bracket_config: Knockout bracket definition required for simulation.
            May be ``None`` when only data-build or training methods are used.
        root: Project root directory.  All relative paths from ``config`` are
            resolved against this path.  Defaults to the current working directory.
    """

    def __init__(
        self,
        config: AppConfig,
        bracket_config: BracketConfig | None = None,
        root: Path | None = None,
    ) -> None:
        self._config = config
        self._bracket_config = bracket_config
        self._root = root or Path.cwd()

    # ------------------------------------------------------------------
    # Data build methods
    # ------------------------------------------------------------------

    def run_build_teams(self, season: str | None = None) -> Path:
        """Fetch team metadata from football-data.org and write teams.csv.

        Args:
            season: Ignored for football (``TeamsBuilder`` queries by competition
                code, not season).  Accepted for API parity with the NBA original.

        Returns:
            Absolute path to the written teams.csv file.
        """
        client = FootballClient(self._config.client)
        builder = TeamsBuilder(client)
        df = builder.build()

        output = self._root / (self._config.data.teams_path or "dataset/csv/teams.csv")
        output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output, index=False)
        logger.info("teams.csv saved to %s (%d rows)", output, len(df))
        return output

    def run_build_matches(self, start_year: int, end_year: int) -> Path:
        """Fetch match results from football-data.org and write matches.csv.

        Mirrors ``Pipeline.run_build_games`` in the NBA original.

        Args:
            start_year: First season start year (e.g. ``2006``).
            end_year: Last season start year (e.g. ``2026``).

        Returns:
            Absolute path to the written matches.csv file.
        """
        client = FootballClient(self._config.client)
        builder = MatchesBuilder(
            client,
            start_year=start_year,
            end_year=end_year,
        )
        df = builder.build()

        output = self._root / (
            self._config.data.matches_path or "dataset/csv/matches.csv"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output, index=False)
        logger.info("matches.csv saved to %s (%d rows)", output, len(df))
        return output

    def run_build_match_details(
        self,
        match_ids: list[str],
        checkpoint_every: int = 100,
    ) -> Path:
        """Fetch per-match detail stats and write match_details.csv.

        Mirrors ``Pipeline.run_build_games_details`` in the NBA original.

        Args:
            match_ids: String match IDs to fetch (e.g. from the MATCH_ID column
                of matches.csv).  Converted to integers internally.
            checkpoint_every: Persist partial results every N matches so that
                interrupted runs can be resumed without re-fetching.

        Returns:
            Absolute path to the written match_details.csv file.
        """
        client = FootballClient(self._config.client)
        int_ids = [int(mid) for mid in match_ids]
        output = self._root / self._config.data.match_details_csv_path

        build_match_details_csv(
            output,
            int_ids,
            client=client,
            checkpoint_every=checkpoint_every,
        )
        logger.info("match_details.csv saved to %s", output)
        return output

    # ------------------------------------------------------------------
    # Clean / train / fit
    # ------------------------------------------------------------------

    def run_clean(self) -> Path:
        """Load raw CSVs, clean, and write train_data.csv.

        Loads matches.csv (and match_details.csv when configured) via
        ``DataLoader``, merges them through ``DataCleaner``, and writes
        the result to the path specified by ``DataConfig.output_path``.

        Returns:
            Absolute path to the written train_data.csv file.
        """
        loader = DataLoader(self._config.data, root=self._root)
        matches_df = loader.load_matches()

        # Load match details when the path is configured and the file exists.
        details_df: pd.DataFrame | None = None
        details_path = self._root / self._config.data.match_details_csv_path
        if details_path.exists():
            details_df = pd.read_csv(details_path)
            assert details_df is not None
            logger.info(
                "Loaded match details from %s (%d rows)", details_path, len(details_df)
            )
        else:
            logger.info(
                "match_details.csv not found at %s — cleaning without detail stats",
                details_path,
            )

        cleaner = DataCleaner(self._config.data, self._config.features)
        output = self._root / self._config.data.output_path
        cleaner.write(matches_df, details_df=details_df, root=self._root)

        logger.info("Cleaned data written to %s", output)
        return output

    def run_train(
        self,
        classifier_names: list[str] | None = None,
    ) -> dict[str, tuple[Classifier, dict[str, Any]]]:
        """Train classifiers and evaluate.

        Args:
            classifier_names: Subset to train.  Defaults to all three:
                ``["svm", "random_forest", "naive_bayes"]``.

        Returns:
            Mapping of ``{name: (fitted_classifier, metrics_dict)}``.
        """
        data_path = self._root / self._config.data.output_path
        df = pd.read_csv(data_path)

        trainer = ClassifierTrainer(self._config.training, self._config.features)
        evaluator = ModelEvaluator()
        X_train, X_test, y_train, y_test = trainer.prepare_data(df)

        names = classifier_names or ["svm", "random_forest", "naive_bayes"]
        results: dict[str, tuple[Classifier, dict[str, Any]]] = {}

        for name in names:
            logger.info("Training %s...", name)
            clf = ClassifierFactory.create(name, self._config.training)
            ClassifierTrainer.train(clf, X_train, y_train)
            metrics = evaluator.evaluate(clf, X_test, y_test)
            results[name] = (clf, metrics)

            model_dir = self._root / "output" / "models"
            ClassifierTrainer.save_model(clf, model_dir / f"{name}.joblib")

        return results

    def run_fit_distributions(self) -> dict[str, list[FittedDistribution]]:
        """Fit statistical distributions per team and persist to JSON.

        Reads train_data.csv, fits one scipy distribution per base statistic
        per team via ``DistributionFitter``, and saves the result to
        ``output/distributions.json``.

        Returns:
            Mapping of ``{team_name: [FittedDistribution per base stat]}``.
        """
        data_path = self._root / self._config.data.output_path
        df = pd.read_csv(data_path)

        fitter = DistributionFitter(self._config.distributions, self._config.features)
        team_dists = fitter.fit_all_teams(df)

        dist_path = self._root / "output" / "distributions.json"
        DistributionFitter.save(team_dists, dist_path)

        return team_dists

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def run_simulate(
        self,
        classifier: Classifier | None = None,
        team_distributions: dict[str, list[FittedDistribution]] | None = None,
        progress_callback: Callable[[int], None] | None = None,
    ) -> dict[int, RoundResult]:
        """Run Monte Carlo knockout-bracket simulation.

        Args:
            classifier: A fitted classifier satisfying the ``Classifier``
                Protocol.  When ``None`` the model is loaded from
                ``output/models/{classifier_name}.joblib``.
            team_distributions: Pre-loaded distribution mapping.  When ``None``
                distributions are loaded from ``output/distributions.json``.
            progress_callback: Optional callable invoked after each simulation
                with the 1-based iteration count (for progress bars).

        Returns:
            ``{round_number: RoundResult}`` with per-team advancement counts and
            probabilities.  Round 0 = Round of 32, 4 = Final.

        Raises:
            ValueError: If no bracket config was provided at construction.
        """
        if self._bracket_config is None:
            raise ValueError("Bracket config is required for simulation")

        if classifier is None:
            clf_name = self._config.simulation.classifier
            model_path = self._root / "output" / "models" / f"{clf_name}.joblib"
            classifier = ClassifierTrainer.load_model(model_path)

        if team_distributions is None:
            dist_path = self._root / "output" / "distributions.json"
            team_distributions = DistributionFitter.load(dist_path)

        sampler = FeatureSampler(self._config.features)
        predictor = GamePredictor(classifier, sampler, team_distributions)
        simulator = TournamentSimulator(predictor, self._config.simulation)

        rounds = simulator.simulate(
            self._bracket_config.matchups,
            self._config.simulation.n_simulations,
            progress_callback=progress_callback,
        )

        return rounds

    # ------------------------------------------------------------------
    # Full pipeline convenience method
    # ------------------------------------------------------------------

    def run_full(
        self,
        progress_callback: Callable[[int], None] | None = None,
    ) -> dict[int, RoundResult]:
        """Execute the entire pipeline: clean -> train -> fit -> simulate -> visualize.

        Args:
            progress_callback: Forwarded to ``run_simulate`` for progress tracking.

        Returns:
            Simulation results as returned by ``run_simulate``.
        """
        from worldcup_playoff.visualization.plots import ResultPlotter

        self.run_clean()
        results = self.run_train()
        team_dists = self.run_fit_distributions()

        clf_name = self._config.simulation.classifier
        classifier, _ = results[clf_name]

        rounds = self.run_simulate(
            classifier=classifier,
            team_distributions=team_dists,
            progress_callback=progress_callback,
        )

        plotter = ResultPlotter(self._config.visualization)
        output_dir = self._root / self._config.visualization.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        plotter.plot_round_probabilities(
            rounds,
            output_path=output_dir / "probabilities.png",
        )

        return rounds
