"""Public API for the worldcup_playoff simulation subpackage.

Forecast engine: Dixon-Coles bivariate-Poisson scoring, the WC2026 knockout
resolver (regulation → extra time → penalties), and the Monte-Carlo live
forecaster (group + knockout).
"""

from __future__ import annotations

from worldcup_playoff.simulation.poisson import (
    DixonColesEstimator,
    ScorelineSampler,
    TeamAbilities,
    blend_abilities_with_elo,
    decay_weight,
    decisive_scoreline,
    dixon_coles_tau,
    fit_dixon_coles,
    lambdas,
    make_sampler,
    modal_scoreline,
    score_matrix,
)
from worldcup_playoff.simulation.knockout import (
    KnockoutSimulator,
    R32_SLOTS,
    RoundResult,
    resolve_r32,
    resolve_tie,
)
from worldcup_playoff.simulation.live_forecast import (
    ForecastResult,
    LiveForecaster,
)

__all__ = [
    "DixonColesEstimator",
    "ForecastResult",
    "KnockoutSimulator",
    "LiveForecaster",
    "R32_SLOTS",
    "RoundResult",
    "ScorelineSampler",
    "TeamAbilities",
    "blend_abilities_with_elo",
    "decay_weight",
    "decisive_scoreline",
    "dixon_coles_tau",
    "fit_dixon_coles",
    "lambdas",
    "make_sampler",
    "modal_scoreline",
    "resolve_r32",
    "resolve_tie",
    "score_matrix",
]
