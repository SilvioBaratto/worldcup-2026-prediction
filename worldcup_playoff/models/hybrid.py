"""Groll-style RF/GBM goal-based hybrid model (primary predictor).

Predicts per-match expected goals via a blended Random-Forest + Gradient-Boosting
regressor pair, then converts the goal pair to a W/D/L distribution via the
reused Dixon-Coles τ-corrected score matrix from ``simulation.poisson``.

Design matrix conventions:
- Explicit allow-list: only numeric + encoded columns enter; targets/identity/metadata
  (home_goals, away_goals, date, home_team, away_team, tournament) are silently excluded.
- Confederation strings (home_confederation, away_confederation) are one-hot encoded
  against the fixed ``CONFEDERATIONS`` tuple — never data-dependent discovery.
- None/unknown confederation → all-zero one-hot row (no NaN leakage).
- Nullable numeric columns (rest_days, ranking) are imputed at fit time with the
  training-set median; sentinels are stored and reused at inference (no leakage).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from worldcup_playoff.config import HybridConfig
from worldcup_playoff.features.confederation import CONFEDERATIONS
from worldcup_playoff.models.dataset import MatchDataset
from worldcup_playoff.simulation.poisson import score_matrix

__all__ = ["GoalPrediction", "HybridModel", "fit_hybrid"]

# ─── Constants ───────────────────────────────────────────────────────────────

_FORBIDDEN: frozenset[str] = frozenset(
    {"home_goals", "away_goals", "date", "home_team", "away_team", "tournament"}
)
_CONFED_COLS: frozenset[str] = frozenset({"home_confederation", "away_confederation"})


# ─── Value object ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GoalPrediction:
    """Outcome distribution and expected goal pair for one match."""

    prob_home: float
    prob_draw: float
    prob_away: float
    home_goals: float
    away_goals: float
    margin: float  # home_goals − away_goals


# ─── Pure module-level helpers ────────────────────────────────────────────────


def _one_hot_confed(series: pd.Series) -> np.ndarray:
    """Encode confederation strings → (n, 6) float array; None/unknown → all-zero row."""
    cats = list(CONFEDERATIONS)
    out = np.zeros((len(series), len(cats)), dtype=float)
    for i, val in enumerate(series):
        if val in cats:
            out[i, cats.index(val)] = 1.0
    return out


def _numeric_col(series: pd.Series, sentinel: float) -> np.ndarray:
    """Cast a numeric-or-bool Series to a (n,1) float array, filling NaN with sentinel."""
    if pd.api.types.is_bool_dtype(series):
        return np.asarray(series.fillna(False).astype(float), dtype=float).reshape(-1, 1)
    coerced = pd.to_numeric(series, errors="coerce").fillna(sentinel)
    return np.asarray(coerced, dtype=float).reshape(-1, 1)


def _col_array(df: pd.DataFrame, col: str, sentinels: dict[str, float]) -> np.ndarray:
    """Convert one feature column to a (n, k) float array; impute NaN via sentinel."""
    if col not in df.columns:
        return np.full((len(df), 1), sentinels.get(col, 0.0))
    series = df[col]
    if col in _CONFED_COLS:
        return _one_hot_confed(series)
    return _numeric_col(series, sentinels.get(col, 0.0))


def build_design_matrix(
    df: pd.DataFrame, feature_cols: list[str], sentinels: dict[str, float] | None = None
) -> np.ndarray:
    """Return float design matrix; excludes forbidden cols, encodes categoricals, imputes NaN."""
    safe = [c for c in feature_cols if c not in _FORBIDDEN]
    if not safe:
        return np.zeros((len(df), 0))
    return np.hstack([_col_array(df, c, sentinels or {}) for c in safe])


def _fit_sentinels(df: pd.DataFrame, cols: list[str]) -> dict[str, float]:
    """Compute fit-time median imputation values for nullable numeric columns."""
    out: dict[str, float] = {}
    for col in cols:
        if col in _FORBIDDEN or col in _CONFED_COLS or col not in df.columns:
            continue
        median = float(pd.to_numeric(df[col], errors="coerce").median())
        out[col] = 0.0 if np.isnan(median) else median
    return out


def derive_outcome_probs(
    lh: float, la: float, *, max_goals: int, rho: float
) -> tuple[float, float, float]:
    """Return (prob_home, prob_draw, prob_away) via the τ-corrected score matrix."""
    mat = score_matrix(lh, la, rho=rho, max_goals=max_goals)
    ph = float(np.tril(mat, -1).sum())  # home score > away score
    pd_ = float(np.diag(mat).sum())  # equal scores (draw)
    pa = float(np.triu(mat, 1).sum())  # away score > home score
    return ph, pd_, pa


def _make_prediction(lh: float, la: float, cfg: HybridConfig) -> GoalPrediction:
    ph, pd_, pa = derive_outcome_probs(lh, la, max_goals=cfg.max_goals, rho=cfg.rho)
    return GoalPrediction(ph, pd_, pa, lh, la, lh - la)


def _new_rf(cfg: HybridConfig) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=cfg.rf_n_estimators,
        max_depth=cfg.rf_max_depth,
        random_state=cfg.random_seed,
        n_jobs=1,
    )


def _new_gb(cfg: HybridConfig) -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        n_estimators=cfg.gb_n_estimators,
        learning_rate=cfg.gb_learning_rate,
        random_state=cfg.random_seed,
    )


def _fit_pair(
    cfg: HybridConfig, X: np.ndarray, y: np.ndarray
) -> tuple[RandomForestRegressor, GradientBoostingRegressor]:
    """Fit and return one (RF, GB) regressor pair for a single goal target."""
    return _new_rf(cfg).fit(X, y), _new_gb(cfg).fit(X, y)


# ─── Thin model class ─────────────────────────────────────────────────────────


class HybridModel:
    """RF+GB blended goal-regression hybrid.  Trains on ``MatchDataset.train`` only."""

    def __init__(self, config: HybridConfig | None = None, *, random_seed: int = 42) -> None:
        self._cfg = config if config is not None else HybridConfig(random_seed=random_seed)
        self._feature_cols: list[str] = []
        self._sentinels: dict[str, float] = {}
        self._rf_h: RandomForestRegressor | None = None
        self._rf_a: RandomForestRegressor | None = None
        self._gb_h: GradientBoostingRegressor | None = None
        self._gb_a: GradientBoostingRegressor | None = None

    def fit(self, dataset: MatchDataset) -> "HybridModel":
        self._feature_cols = dataset.feature_cols
        self._sentinels = _fit_sentinels(dataset.train, self._feature_cols)
        X = build_design_matrix(dataset.train, self._feature_cols, self._sentinels)
        y_h = dataset.train["home_goals"].astype(float).values
        y_a = dataset.train["away_goals"].astype(float).values
        self._rf_h, self._gb_h = _fit_pair(self._cfg, X, y_h)
        self._rf_a, self._gb_a = _fit_pair(self._cfg, X, y_a)
        return self

    def predict_goals(self, features: dict[str, Any]) -> tuple[float, float]:
        X = build_design_matrix(pd.DataFrame([features]), self._feature_cols, self._sentinels)
        lh = max(0.0, float((self._rf_h.predict(X)[0] + self._gb_h.predict(X)[0]) / 2))  # type: ignore[union-attr]
        la = max(0.0, float((self._rf_a.predict(X)[0] + self._gb_a.predict(X)[0]) / 2))  # type: ignore[union-attr]
        return lh, la

    def predict(self, features: dict[str, Any]) -> GoalPrediction:
        lh, la = self.predict_goals(features)
        return _make_prediction(lh, la, self._cfg)


# ─── Factory ──────────────────────────────────────────────────────────────────


def fit_hybrid(dataset: MatchDataset, config: HybridConfig | None = None) -> HybridModel:
    """Fit and return a HybridModel from a pre-split MatchDataset."""
    return HybridModel(config=config).fit(dataset)


# ─── High-level CLI entry ─────────────────────────────────────────────────────


def _load_features(root: Any, features_path: str = "dataset/features.csv") -> "pd.DataFrame | None":
    from pathlib import Path
    try:
        return pd.read_csv(Path(root) / features_path)
    except (FileNotFoundError, OSError):
        return None


def _make_dataset(features: "pd.DataFrame", test_size: float) -> MatchDataset:
    from worldcup_playoff.models.dataset import build_dataset
    from worldcup_playoff.features.build import FEATURE_COLUMNS
    feat_cols = [c for c in FEATURE_COLUMNS if c in features.columns]
    return build_dataset(features, test_size, feat_cols)


def train_hybrid(cfg: Any = None, root: Any = None) -> HybridModel | None:
    """High-level CLI entry: load features from disk and fit a HybridModel.

    Returns ``None`` when ``dataset/features.csv`` is unavailable.
    """
    from worldcup_playoff.config import AppConfig
    resolved = cfg if cfg is not None else AppConfig()
    features = _load_features(root or ".")
    if features is None:
        return None
    return fit_hybrid(_make_dataset(features, resolved.hybrid.test_size), resolved.hybrid)
