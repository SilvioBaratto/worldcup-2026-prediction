"""Elo-diff ordered logit — secondary/fallback W/D/L predictor.

Lightweight safety net when the goal hybrid cannot fit (e.g. too few
played rows). Predicts W/D/L probabilities only — no goal output, so it
does not feed goal-difference tiebreaks.

Lazy-imports ``OrderedModel`` inside ``OrderedLogitModel.fit`` so this
module loads fast at collection time even when statsmodels is absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from worldcup_playoff.config import OrderedLogitConfig
from worldcup_playoff.models.dataset import MatchDataset

__all__ = ["OutcomeProbabilities", "OrderedLogitModel", "fit_ordered_logit"]


# ─── Value object ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class OutcomeProbabilities:
    """Calibrated W/D/L probability triple for one match."""

    prob_home: float
    prob_draw: float
    prob_away: float


# ─── Pure module-level helpers ────────────────────────────────────────────────


def _endog(train: pd.DataFrame) -> np.ndarray:
    """Cast nullable Int64 y_outcome to plain int64 array (safe after played_only)."""
    return np.asarray(train["y_outcome"].astype(int))


def _exog(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Select features as float64 array; no constant prepended (OrderedModel uses cut-points)."""
    return np.asarray(df[features].astype(float), dtype=float)


def _to_outcome(row: np.ndarray) -> OutcomeProbabilities:
    """Map an [away, draw, home] probability row to the value object."""
    return OutcomeProbabilities(
        prob_away=float(row[0]),
        prob_draw=float(row[1]),
        prob_home=float(row[2]),
    )


# ─── Model class ─────────────────────────────────────────────────────────────


class OrderedLogitModel:
    """Ordered logit over W/D/L outcomes; trains on MatchDataset.train only.

    Secondary/fallback predictor — predicts W/D/L, not goals.
    """

    def __init__(self, config: OrderedLogitConfig | None = None) -> None:
        self._cfg = config or OrderedLogitConfig()
        self._result: Any = None  # statsmodels OrderedResultsWrapper post-fit

    def fit(self, dataset: MatchDataset) -> "OrderedLogitModel":
        """Fit on dataset.train only — time-aware, no internal split/shuffle."""
        from statsmodels.miscmodels.ordinal_model import OrderedModel  # lazy import

        train = dataset.train
        self._result = OrderedModel(
            _endog(train), _exog(train, self._cfg.features), distr="logit"
        ).fit(method="bfgs", maxiter=self._cfg.maxiter, disp=False)
        return self

    def predict(self, df: pd.DataFrame) -> list[OutcomeProbabilities]:
        """Return OutcomeProbabilities for each row of df; no API key required."""
        probs: np.ndarray = self._result.predict(exog=_exog(df, self._cfg.features))
        return [_to_outcome(row) for row in probs]


# ─── Factory ──────────────────────────────────────────────────────────────────


def fit_ordered_logit(
    dataset: MatchDataset, config: OrderedLogitConfig | None = None
) -> OrderedLogitModel:
    """Fit and return an OrderedLogitModel from a pre-split MatchDataset."""
    return OrderedLogitModel(config).fit(dataset)
