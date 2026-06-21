"""Tests for ClassifierFactory and ClassifierTrainer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from worldcup_playoff.config import FeaturesConfig, TrainingConfig
from worldcup_playoff.models.classifiers import ClassifierFactory, ClassifierTrainer


# ---------------------------------------------------------------------------
# ClassifierFactory
# ---------------------------------------------------------------------------


class TestClassifierFactory:
    def test_create_svm_returns_pipeline(self) -> None:
        """SVM must be wrapped in a Pipeline for feature scaling."""
        clf = ClassifierFactory.create("svm", TrainingConfig())
        assert isinstance(clf, Pipeline)

    def test_create_svm_pipeline_has_standard_scaler(self) -> None:
        clf = ClassifierFactory.create("svm", TrainingConfig())
        scaler = clf.named_steps["standardscaler"]
        assert isinstance(scaler, StandardScaler)

    def test_create_svm_pipeline_has_svc_with_correct_params(self) -> None:
        clf = ClassifierFactory.create("svm", TrainingConfig())
        svc = clf.named_steps["svc"]
        assert isinstance(svc, SVC)
        assert svc.C == 0.1
        assert svc.kernel == "linear"
        assert svc.probability is True

    def test_create_svm_fits_and_predicts(self) -> None:
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 100, (80, 10))
        y = rng.integers(0, 2, 80)
        clf = ClassifierFactory.create("svm", TrainingConfig())
        clf.fit(X, y)
        preds = clf.predict(X)
        assert len(preds) == 80
        assert set(preds).issubset({0, 1})

    def test_create_random_forest(self) -> None:
        clf = ClassifierFactory.create("random_forest", TrainingConfig())
        assert isinstance(clf, RandomForestClassifier)
        assert clf.n_estimators == 500
        assert clf.bootstrap is True

    def test_create_random_forest_with_custom_config(self) -> None:
        from worldcup_playoff.config import RandomForestConfig
        cfg = TrainingConfig(random_forest=RandomForestConfig(n_estimators=100, max_depth=10))
        clf = ClassifierFactory.create("random_forest", cfg)
        assert clf.n_estimators == 100
        assert clf.max_depth == 10

    def test_create_naive_bayes(self) -> None:
        clf = ClassifierFactory.create("naive_bayes", TrainingConfig())
        assert isinstance(clf, GaussianNB)

    def test_create_unknown_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown classifier"):
            ClassifierFactory.create("catboost", TrainingConfig())


# ---------------------------------------------------------------------------
# ClassifierTrainer.prepare_data
# ---------------------------------------------------------------------------


def _make_chronological_df(n: int = 100) -> pd.DataFrame:
    """DataFrame with monotonically increasing feature values (simulates time ordering)."""
    features = FeaturesConfig()
    data: dict[str, object] = {
        col: np.arange(n, dtype=float) for col in features.selected
    }
    data["HOME_WIN"] = (np.arange(n) % 2).astype(int)
    return pd.DataFrame(data)


class TestClassifierTrainer:
    def test_prepare_data_preserves_temporal_order(self) -> None:
        """Train rows must all precede test rows — no temporal leakage."""
        n = 100
        df = _make_chronological_df(n)
        trainer = ClassifierTrainer(TrainingConfig(test_size=0.3), FeaturesConfig())
        X_train, X_test, _, _ = trainer.prepare_data(df)

        train_max = float(X_train[:, 0].max())
        test_min = float(X_test[:, 0].min())
        assert train_max < test_min, (
            f"Temporal leakage: latest train sample ({train_max}) >= "
            f"earliest test sample ({test_min}). shuffle=False must be set."
        )

    def test_prepare_data_split_sizes(self) -> None:
        n = 100
        df = _make_chronological_df(n)
        trainer = ClassifierTrainer(TrainingConfig(test_size=0.3), FeaturesConfig())
        X_train, X_test, y_train, y_test = trainer.prepare_data(df)
        assert len(X_train) + len(X_test) == n
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        assert len(X_test) == pytest.approx(n * 0.3, abs=1)

    def test_prepare_data_reads_home_win_column(self) -> None:
        """HOME_WIN column must be used as target, not any other column."""
        df = _make_chronological_df(50)
        df["HOME_WIN"] = 1  # all home wins
        trainer = ClassifierTrainer(TrainingConfig(test_size=0.3), FeaturesConfig())
        _, _, y_train, y_test = trainer.prepare_data(df)
        assert all(y_train == 1)
        assert all(y_test == 1)

    def test_train_fits_classifier(self) -> None:
        X = np.random.rand(60, 10)
        y = np.random.choice([0, 1], 60)
        clf = GaussianNB()
        ClassifierTrainer.train(clf, X, y)
        preds = clf.predict(X)
        assert len(preds) == 60

    def test_save_and_load_model_round_trip(self, tmp_path: Path) -> None:
        """Model saved with save_model must be loadable and produce identical predictions."""
        X = np.random.rand(60, 10)
        y = np.random.choice([0, 1], 60)
        clf = GaussianNB()
        clf.fit(X, y)
        original_preds = clf.predict(X)

        path = tmp_path / "naive_bayes.joblib"
        ClassifierTrainer.save_model(clf, path)
        assert path.exists()

        loaded = ClassifierTrainer.load_model(path)
        loaded_preds = loaded.predict(X)
        np.testing.assert_array_equal(original_preds, loaded_preds)

    def test_save_model_creates_parent_directories(self, tmp_path: Path) -> None:
        clf = GaussianNB()
        clf.fit(np.random.rand(10, 10), np.zeros(10, dtype=int))
        path = tmp_path / "output" / "models" / "test.joblib"
        ClassifierTrainer.save_model(clf, path)
        assert path.exists()
