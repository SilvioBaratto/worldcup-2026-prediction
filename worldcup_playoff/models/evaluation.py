"""Model evaluation and comparison.

Public API
----------
Pure metric functions (source-blind, no sklearn dependency):
  rank_probability_score  — ordered 3-class RPS (primary)
  multiclass_log_loss     — mean cross-entropy
  brier_score             — mean multi-class Brier score

Orchestration:
  backtest_hybrid         — time-aware WC backtest vs bookmaker + legacy baselines

Legacy (kept for pipeline compatibility):
  ModelEvaluator          — confusion matrix + classification report
  plot_roc_curves         — ROC curve visualisation
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
)

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

_EPS: float = 1e-15
_DEFAULT_WC_YEARS: list[int] = [2014, 2018, 2022]
_META_COLS: frozenset[str] = frozenset(
    {"date", "home_team", "away_team", "tournament", "outcome"}
)

# ── Pure metric functions ────────────────────────────────────────────────────


def rank_probability_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean RPS for ordered 3-class (Win/Draw/Loss) predictions.

    RPS_i = (1/(J-1)) * Σ_{k=0}^{J-2} (CDF_pred_{i,k} − CDF_obs_{i,k})²
    averaged over N matches (J=3 → divides by 2).
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred, dtype=float)
    cdf_pred = np.cumsum(yp, axis=1)[:, :-1]          # (N, J-1)
    cutpoints = np.arange(yp.shape[1] - 1)            # [0, 1]
    cdf_obs = (yt[:, np.newaxis] <= cutpoints).astype(float)
    return float(np.mean(np.mean((cdf_pred - cdf_obs) ** 2, axis=1)))


def multiclass_log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean cross-entropy loss; clips probabilities to avoid log(0)."""
    yt = np.asarray(y_true)
    yp = np.clip(np.asarray(y_pred, dtype=float), _EPS, 1.0)
    return float(-np.mean(np.log(yp[np.arange(len(yt)), yt])))


def brier_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean multi-class Brier score (mean squared error in probability space)."""
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred, dtype=float)
    one_hot = np.eye(yp.shape[1])[yt]
    return float(np.mean(np.sum((yp - one_hot) ** 2, axis=1)))


# ── Backtest helpers ─────────────────────────────────────────────────────────


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _META_COLS]


def _slice_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    return df[pd.to_datetime(df["date"]).dt.year == year].reset_index(drop=True)


def _pre_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    return df[pd.to_datetime(df["date"]).dt.year < year].reset_index(drop=True)


def _bookmaker_rps(year_matches: pd.DataFrame, odds: pd.DataFrame) -> float | None:
    """Merge bookmaker odds onto year matches and return mean RPS; None if no overlap."""
    key = ["date", "home_team", "away_team"]
    merged = year_matches[key + ["outcome"]].merge(
        odds[key + ["p_win", "p_draw", "p_loss"]], on=key, how="inner"
    )
    if merged.empty:
        return None
    return rank_probability_score(
        merged["outcome"].to_numpy(),
        merged[["p_win", "p_draw", "p_loss"]].to_numpy(),
    )


def _build_row(
    year_matches: pd.DataFrame,
    hybrid: Any,
    legacy: Any,
    feat_cols: list[str],
    odds: pd.DataFrame | None,
) -> dict[str, Any]:
    X = year_matches[feat_cols].to_numpy()
    y_true = year_matches["outcome"].to_numpy()
    row: dict[str, Any] = {
        "rps_hybrid": rank_probability_score(y_true, hybrid.predict_proba(X)),
        "rps_legacy": rank_probability_score(y_true, legacy.predict_proba(X)),
    }
    if odds is not None:
        bk = _bookmaker_rps(year_matches, odds)
        if bk is not None:
            row["rps_bookmaker"] = bk
    return row


def backtest_hybrid(
    matches: pd.DataFrame,
    hybrid: Any,
    legacy: Any,
    odds: pd.DataFrame | None = None,
    years: list[int] | None = None,
) -> pd.DataFrame:
    """Time-aware WC backtest: train on pre-year data, evaluate on WC year.

    Parameters
    ----------
    matches:  DataFrame with 'date', 'outcome', and feature columns.
    hybrid:   Model with fit(X, y) and predict_proba(X) → (N, 3).
    legacy:   Same protocol as hybrid (used as second baseline).
    odds:     Optional bookmaker odds with 'date', 'home_team', 'away_team',
              'p_win', 'p_draw', 'p_loss'. Skipped when None.
    years:    WC years to backtest (default [2014, 2018, 2022]).

    Returns
    -------
    DataFrame indexed by year with columns rps_hybrid, rps_legacy, and
    (when odds are not None) rps_bookmaker.
    """
    eval_years = years if years is not None else _DEFAULT_WC_YEARS
    feat_cols = _feature_cols(matches)
    rows: dict[int, dict[str, Any]] = {}
    for year in eval_years:
        year_matches = _slice_year(matches, year)
        if year_matches.empty:
            continue
        train = _pre_year(matches, year)
        if not train.empty:
            X_tr = train[feat_cols].to_numpy()
            y_tr = train["outcome"].to_numpy()
            hybrid.fit(X_tr, y_tr)
            legacy.fit(X_tr, y_tr)
        rows[year] = _build_row(year_matches, hybrid, legacy, feat_cols, odds)
    result = pd.DataFrame.from_dict(rows, orient="index")
    result.index.name = "year"
    return result


# ── High-level CLI entry ─────────────────────────────────────────────────────


def _read_features(root: Any) -> "pd.DataFrame | None":
    from pathlib import Path
    try:
        df = pd.read_csv(Path(root) / "dataset" / "features.csv")
        return df if "outcome" in df.columns else None
    except (FileNotFoundError, OSError):
        return None


def _sklearn_pair(seed: int = 42) -> tuple[Any, Any]:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    return (
        GradientBoostingClassifier(random_state=seed),
        RandomForestClassifier(n_estimators=10, random_state=seed),
    )


def _build_odds_frame(cfg: Any, odds_mod: Any) -> "pd.DataFrame | None":
    """Load and concat match odds for all configured seasons; None when all empty/blocked."""
    seasons = getattr(getattr(cfg, "odds", None), "seasons", [])
    frames: list[pd.DataFrame] = []
    for year in seasons:
        try:
            raw = odds_mod.load_match_odds(f"wc{year}", cfg)
            if not raw.empty:
                frames.append(raw)
        except Exception:
            logger.warning("Match odds unavailable for wc%d; skipping.", year)
    return pd.concat(frames, ignore_index=True) if frames else None


def run_backtest(cfg: Any = None, root: Any = None) -> "pd.DataFrame | None":
    """High-level CLI entry: load features+targets and run time-aware WC backtest.

    Loads bookmaker match odds for all seasons in cfg.odds.seasons (via
    odds.load_match_odds) and passes them to backtest_hybrid so rps_bookmaker is
    included when odds are available.  Passes odds=None when all scrapers are
    blocked, so backtest_hybrid degrades gracefully.

    Returns None when dataset/features.csv is unavailable or lacks an outcome column.
    """
    from worldcup_playoff.config import AppConfig
    from worldcup_playoff.data import odds as _odds_mod

    resolved = cfg if cfg is not None else AppConfig()
    matches = _read_features(root or ".")
    if matches is None:
        return None
    seed = getattr(getattr(resolved, "hybrid", None), "random_seed", 42)
    combined_odds = _build_odds_frame(resolved, _odds_mod)
    return backtest_hybrid(matches, *_sklearn_pair(seed), odds=combined_odds)


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
