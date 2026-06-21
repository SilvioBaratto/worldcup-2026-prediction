"""Knockout tie prediction using sampled features and a trained classifier.

Mirrors ``nba_playoff.simulation.game`` but replaces the best-of-N series loop
with a **single-match** tie: one feature vector is sampled, the classifier runs
once, and the result immediately determines the advancing team.

In the World Cup knockout stage extra time and penalties collapse into a single
win/loss outcome, so there is no ``num_games`` parameter here — the method is
``predict_tie(home, away) -> str``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from worldcup_playoff.simulation.distributions import (
    FeatureSampler,
    FittedDistribution,
)


class GamePredictor:
    """Predicts the winner of a single knockout tie between two national teams.

    Uses dependency injection instead of global state:

    - ``classifier``: a trained ML model satisfying the ``Classifier`` Protocol.
    - ``sampler``: a ``FeatureSampler`` that draws synthetic match statistics.
    - ``team_distributions``: fitted distributions keyed by country name.

    The NBA ``GamePredictor.predict`` accepted ``num_games`` for a best-of-N
    series.  Here ``predict_tie`` always simulates exactly one match — the World
    Cup knockout format leaves no room for series length configuration.
    """

    def __init__(
        self,
        classifier: Any,
        sampler: FeatureSampler,
        team_distributions: dict[str, list[FittedDistribution]],
    ) -> None:
        self._classifier = classifier
        self._sampler = sampler
        self._team_distributions = team_distributions

    def predict_tie(self, home: str, away: str) -> str:
        """Simulate a single knockout tie and return the advancing team.

        Samples one feature vector from the fitted distributions of *home* and
        *away*, concatenates them in the order prescribed by
        ``FeaturesConfig.selected`` (home stats first, away stats second), feeds
        the vector to the classifier, and maps the binary prediction back to a
        team name.

        Args:
            home: Country name of the team playing at (or designated as) home.
            away: Country name of the team playing away.

        Returns:
            The country name of the winning team (either *home* or *away*).

        Raises:
            KeyError: If *home* or *away* are not present in
                ``team_distributions``.
        """
        home_features = self._sampler.sample(
            self._team_distributions[home], size=1
        )
        away_features = self._sampler.sample(
            self._team_distributions[away], size=1
        )
        combined: np.ndarray = np.hstack([home_features, away_features])
        # predict returns an array of shape (1,); 1 = home win, 0 = away win.
        prediction: int = int(self._classifier.predict(combined)[0])
        return home if prediction == 1 else away
