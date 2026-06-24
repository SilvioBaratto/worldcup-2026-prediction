from worldcup_playoff.models.classifiers import ClassifierFactory, ClassifierTrainer
from worldcup_playoff.models.dataset import MatchDataset, build_dataset
from worldcup_playoff.models.evaluation import ModelEvaluator
from worldcup_playoff.models.hybrid import GoalPrediction, HybridModel, fit_hybrid
from worldcup_playoff.models.ordered_logit import (
    OutcomeProbabilities,
    OrderedLogitModel,
    fit_ordered_logit,
)

# Alias for callers expecting the full class name from the issue spec.
HybridGoalModel = HybridModel

__all__ = [
    "ClassifierFactory",
    "ClassifierTrainer",
    "GoalPrediction",
    "HybridGoalModel",
    "HybridModel",
    "MatchDataset",
    "ModelEvaluator",
    "OrderedLogitModel",
    "OutcomeProbabilities",
    "build_dataset",
    "fit_hybrid",
    "fit_ordered_logit",
]
