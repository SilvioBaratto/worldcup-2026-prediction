"""Public API for the worldcup_playoff simulation subpackage.

Exports all types and classes needed by the pipeline and CLI layers:

- ``FittedDistribution`` — immutable record of a fitted scipy distribution.
- ``DistributionFitter`` — fits and serializes per-team distributions.
- ``FeatureSampler`` — assembles synthetic feature vectors from distributions.
- ``GamePredictor`` — predicts the winner of a single knockout tie.
- ``TournamentSimulator`` — Monte Carlo bracket runner.
- ``RoundResult`` — per-round advancement counts and probabilities.
- ``BracketSlot`` — node in the bracket tree used for visualization.
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
from worldcup_playoff.simulation.tournament import (
    BracketSlot,
    RoundResult,
    TournamentSimulator,
    build_bracket_tree,
    extract_bracket_slots,
)

__all__ = [
    "BracketSlot",
    "DixonColesEstimator",
    "DistributionFitter",
    "FeatureSampler",
    "FittedDistribution",
    "GamePredictor",
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
    "score_matrix",
]
