"""Model evaluation and comparison."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluates classifier performance."""

    @staticmethod
    def evaluate(
        classifier: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict[str, Any]:
        """Return confusion matrix and classification report as a dict."""
        y_pred = classifier.predict(X_test)
        return {
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "classification_report": classification_report(
                y_test, y_pred, output_dict=True
            ),
        }

    @staticmethod
    def plot_roc_curves(
        classifiers: dict[str, Any],
        X_test: np.ndarray,
        y_test: np.ndarray,
        output_path: Path | None = None,
    ) -> None:
        """Plot ROC curves for multiple classifiers."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 8))
        linestyles = ["-", ":", "--", "-.", (0, (3, 1, 1, 1))]

        for i, (name, clf) in enumerate(classifiers.items()):
            RocCurveDisplay.from_estimator(
                clf, X_test, y_test,
                ax=ax,
                linewidth=3,
                linestyle=linestyles[i % len(linestyles)],
                name=name,
            )

        ax.tick_params(axis="both", labelsize=16)
        ax.set_xlabel("False Positive Rate", fontsize=18)
        ax.set_ylabel("True Positive Rate", fontsize=18)
        ax.legend(fontsize=14)

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, bbox_inches="tight")
            logger.info("ROC curve saved to %s", output_path)
        plt.close(fig)
