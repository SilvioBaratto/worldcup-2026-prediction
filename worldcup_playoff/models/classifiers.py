"""Classifier creation, training, and persistence."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from worldcup_playoff.config import FeaturesConfig, NaiveBayesConfig, RandomForestConfig, SVMConfig, TrainingConfig

logger = logging.getLogger(__name__)


class ClassifierFactory:
    """Creates configured classifier instances from config."""

    @staticmethod
    def create(name: str, config: TrainingConfig) -> Any:
        cfg: SVMConfig | RandomForestConfig | NaiveBayesConfig
        match name:
            case "svm":
                cfg = config.svm
                return make_pipeline(
                    StandardScaler(),
                    SVC(C=cfg.C, gamma=cfg.gamma, kernel=cfg.kernel, probability=True),
                )
            case "random_forest":
                cfg = config.random_forest
                return RandomForestClassifier(
                    n_estimators=cfg.n_estimators,
                    max_features=cfg.max_features,
                    max_depth=cfg.max_depth,
                    bootstrap=cfg.bootstrap,
                )
            case "naive_bayes":
                cfg = config.naive_bayes
                return GaussianNB(var_smoothing=cfg.var_smoothing)
            case _:
                raise ValueError(
                    f"Unknown classifier '{name}'. "
                    f"Choose from: svm, random_forest, naive_bayes"
                )


class ClassifierTrainer:
    """Handles data preparation and model fitting."""

    def __init__(self, config: TrainingConfig, features_config: FeaturesConfig) -> None:
        self._config = config
        self._features = features_config

    def prepare_data(
        self, df: pd.DataFrame
    ) -> tuple[Any, Any, Any, Any]:
        """Extract features/labels and split into train/test.

        Returns (X_train, X_test, y_train, y_test) as numpy arrays.
        """
        X = df[self._features.selected].to_numpy()
        y = df["HOME_WIN"].to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self._config.test_size,
            random_state=self._config.random_state,
            shuffle=False,
        )
        logger.info(
            "Data split: train=%d, test=%d", len(X_train), len(X_test)
        )
        return X_train, X_test, y_train, y_test

    @staticmethod
    def train(
        classifier: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> Any:
        """Fit the classifier on training data."""
        classifier.fit(X_train, y_train)
        return classifier

    @staticmethod
    def save_model(classifier: Any, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(classifier, path)
        logger.info("Model saved to %s", path)

    @staticmethod
    def load_model(path: Path) -> Any:
        logger.info("Loading model from %s", path)
        return joblib.load(path)
