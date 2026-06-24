"""Public API for the worldcup_playoff simulation subpackage.

Exports all types and classes needed by the pipeline and CLI layers:

- ``FittedDistribution`` — immutable record of a fitted scipy distribution.
- ``DistributionFitter`` — fits and serializes per-team distributions.
- ``FeatureSampler`` — assembles synthetic feature vectors from distributions.
- ``GamePredictor`` — predicts the winner of a single knockout tie.
- ``TournamentSimulator`` — Monte Carlo bracket runner (legacy classifier path).
- ``KnockoutSimulator`` — Monte Carlo knockout runner (Poisson + ET + penalty).
- ``RoundResult`` — per-round advancement counts and probabilities.
- ``BracketSlot`` — node in the bracket tree used for visualization.
- ``resolve_tie`` — resolve a single knockout tie (regulation → ET → coin-flip).
- ``R32_SLOTS`` / ``resolve_r32`` — WC2026 R32 bracket template and seeder.
- ``ForecastResult`` — title odds + per-round advancement from the live forecaster.
- ``LiveForecaster`` — Monte Carlo live-forecast orchestrator (group + knockout).
"""

from __future__ import annotations

from worldcup_playoff.simulation.distributions import (
    DistributionFitter,
    FeatureSampler,
    FittedDistribution,
)
from worldcup_playoff.simulation.game import GamePredictor
from worldcup_playoff.simulation.poisson import (
    DixonColesEstimator,
    ScorelineSampler,
    TeamAbilities,
    decay_weight,
    dixon_coles_tau,
    fit_dixon_coles,
    lambdas,
    make_sampler,
    score_matrix,
)
from worldcup_playoff.simulation.knockout import (
    KnockoutSimulator,
    R32_SLOTS,
    resolve_r32,
    resolve_tie,
)
from worldcup_playoff.simulation.live_forecast import (
    ForecastResult,
    LiveForecaster,
)
from worldcup_playoff.simulation.tournament import (
    BracketSlot,
    RoundResult,
    TournamentSimulator,
    build_bracket_tree,
    extract_bracket_slots,
)

__all__ = [
    "BracketSlot",
    "ForecastResult",
    "LiveForecaster",
    "DixonColesEstimator",
    "DistributionFitter",
    "FeatureSampler",
    "FittedDistribution",
    "GamePredictor",
    "KnockoutSimulator",
    "R32_SLOTS",
    "RoundResult",
    "ScorelineSampler",
    "TeamAbilities",
    "TournamentSimulator",
    "build_bracket_tree",
    "decay_weight",
    "dixon_coles_tau",
    "extract_bracket_slots",
    "fit_dixon_coles",
    "lambdas",
    "make_sampler",
    "resolve_r32",
    "resolve_tie",
    "score_matrix",
]
