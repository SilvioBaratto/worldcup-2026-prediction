"""Models subpackage: match dataset, RF/GBM hybrid goal model, and the
backtest evaluator used to calibrate the Elo-prior blend weight."""

from worldcup_playoff.models.dataset import MatchDataset, build_dataset
from worldcup_playoff.models.evaluation import ModelEvaluator
from worldcup_playoff.models.hybrid import GoalPrediction, HybridModel, fit_hybrid

# Alias for callers expecting the full class name from the issue spec.
HybridGoalModel = HybridModel

__all__ = [
    "GoalPrediction",
    "HybridGoalModel",
    "HybridModel",
    "MatchDataset",
    "ModelEvaluator",
    "build_dataset",
    "fit_hybrid",
]
