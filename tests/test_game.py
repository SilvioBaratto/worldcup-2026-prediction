"""Tests for GamePredictor — single-match knockout tie prediction."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from worldcup_playoff.simulation.distributions import (
    FeatureSampler,
    FittedDistribution,
)
from worldcup_playoff.simulation.game import GamePredictor
from worldcup_playoff.config import FeaturesConfig


# ---------------------------------------------------------------------------
# Fake classifiers
# ---------------------------------------------------------------------------


class _AlwaysHomeClassifier:
    """Always predicts home win (1)."""

    def fit(self, X: Any, y: Any) -> "_AlwaysHomeClassifier":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.ones(X.shape[0], dtype=int)


class _AlwaysAwayClassifier:
    """Always predicts away win (0)."""

    def fit(self, X: Any, y: Any) -> "_AlwaysAwayClassifier":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(X.shape[0], dtype=int)


# ---------------------------------------------------------------------------
# GamePredictor.predict_tie
# ---------------------------------------------------------------------------


class TestGamePredictor:
    def _make_predictor(
        self,
        classifier: Any,
        distributions: dict[str, list[FittedDistribution]],
    ) -> GamePredictor:
        sampler = FeatureSampler(FeaturesConfig())
        return GamePredictor(classifier, sampler, distributions)

    def test_predict_tie_returns_one_of_the_two_teams(
        self, sample_distributions: dict[str, list[FittedDistribution]]
    ) -> None:
        predictor = self._make_predictor(_AlwaysHomeClassifier(), sample_distributions)
        winner = predictor.predict_tie("Brazil", "France")
        assert winner in {"Brazil", "France"}

    def test_predict_tie_returns_home_when_classifier_predicts_1(
        self, sample_distributions: dict[str, list[FittedDistribution]]
    ) -> None:
        predictor = self._make_predictor(_AlwaysHomeClassifier(), sample_distributions)
        winner = predictor.predict_tie("Brazil", "France")
        assert winner == "Brazil"

    def test_predict_tie_returns_away_when_classifier_predicts_0(
        self, sample_distributions: dict[str, list[FittedDistribution]]
    ) -> None:
        predictor = self._make_predictor(_AlwaysAwayClassifier(), sample_distributions)
        winner = predictor.predict_tie("Brazil", "France")
        assert winner == "France"

    def test_predict_tie_is_deterministic_with_fixed_classifier(
        self, sample_distributions: dict[str, list[FittedDistribution]]
    ) -> None:
        """Always-home classifier must produce the same result every call."""
        predictor = self._make_predictor(_AlwaysHomeClassifier(), sample_distributions)
        results = [predictor.predict_tie("Germany", "Argentina") for _ in range(10)]
        assert all(r == "Germany" for r in results)

    def test_predict_tie_raises_for_unknown_team(
        self, sample_distributions: dict[str, list[FittedDistribution]]
    ) -> None:
        """Unknown team name must raise KeyError from the distributions lookup."""
        predictor = self._make_predictor(_AlwaysHomeClassifier(), sample_distributions)
        with pytest.raises(KeyError):
            predictor.predict_tie("Brazil", "UnknownTeam")

    def test_predict_tie_with_all_four_team_pairs(
        self, sample_distributions: dict[str, list[FittedDistribution]]
    ) -> None:
        """All known teams must be playable without error."""
        predictor = self._make_predictor(_AlwaysHomeClassifier(), sample_distributions)
        teams = list(sample_distributions.keys())
        for i in range(len(teams)):
            for j in range(len(teams)):
                if i != j:
                    winner = predictor.predict_tie(teams[i], teams[j])
                    assert winner in {teams[i], teams[j]}

    def test_predict_tie_uses_10_feature_vector(
        self, sample_distributions: dict[str, list[FittedDistribution]]
    ) -> None:
        """The classifier must receive a (1, 10) feature matrix."""
        captured: list[np.ndarray] = []

        class _CapturingClassifier:
            def predict(self, X: np.ndarray) -> np.ndarray:
                captured.append(X.copy())
                return np.ones(X.shape[0], dtype=int)

        predictor = self._make_predictor(_CapturingClassifier(), sample_distributions)
        predictor.predict_tie("Brazil", "France")
        assert len(captured) == 1
        assert captured[0].shape == (1, 10)
