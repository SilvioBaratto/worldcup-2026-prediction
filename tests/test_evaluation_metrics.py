"""
Source-blind example + property tests for probabilistic evaluation metrics:
  rank_probability_score, multiclass_log_loss, brier_score.

All expected values are hand-computed directly from the metric formulae —
no implementation source was read during authoring.

Metric formulae (J = 3 ordered classes: 0=Win, 1=Draw, 2=Loss):
  RPS  = (1/(J-1)) * sum_{k=1}^{J-1} (CDF_pred_k - CDF_obs_k)^2   [J-1 = 2]
  LL   = -(1/N) * sum_i  y_i[outcome_i] * log(p_i[outcome_i])
  BS   = (1/N) * sum_i sum_j (p_ij - o_ij)^2

Outcome encoding: 0 = home win (W), 1 = draw (D), 2 = away win / loss (L).
"""

import math

import numpy as np
import pytest
from hypothesis import given, strategies as st

from worldcup_playoff.models.evaluation import (
    brier_score,
    multiclass_log_loss,
    rank_probability_score,
)


# ---------------------------------------------------------------------------
# rank_probability_score — example tests
# ---------------------------------------------------------------------------


def test_when_win_is_predicted_with_certainty_and_outcome_is_win_then_rps_is_zero():
    """Perfect prediction of Win → RPS = 0."""
    probs = np.array([[1.0, 0.0, 0.0]])
    y_true = np.array([0])
    assert rank_probability_score(y_true, probs) == pytest.approx(0.0)


def test_when_draw_is_predicted_with_certainty_and_outcome_is_draw_then_rps_is_zero():
    """Perfect prediction of Draw → RPS = 0.
    CDF_pred=[0,1], CDF_obs=[0,1] → both squared diffs are 0.
    """
    probs = np.array([[0.0, 1.0, 0.0]])
    y_true = np.array([1])
    assert rank_probability_score(y_true, probs) == pytest.approx(0.0)


def test_when_win_is_predicted_with_certainty_but_outcome_is_loss_then_rps_is_one():
    """
    Worst-case ordered-RPS: predict Win with certainty, actual outcome is Loss.
    CDF_pred = [1, 1]   CDF_obs = [0, 0]
    RPS = 0.5 * (1^2 + 1^2) = 1.0
    """
    probs = np.array([[1.0, 0.0, 0.0]])
    y_true = np.array([2])
    assert rank_probability_score(y_true, probs) == pytest.approx(1.0)


def test_when_uniform_prediction_and_win_outcome_then_rps_equals_five_eighteenths():
    """
    p = [1/3, 1/3, 1/3], outcome = Win (0).
    CDF_pred = [1/3, 2/3]   CDF_obs = [1, 1]
    RPS = 0.5 * ((1/3 - 1)^2 + (2/3 - 1)^2)
        = 0.5 * (4/9 + 1/9)
        = 5/18  ≈  0.27778
    """
    probs = np.array([[1 / 3, 1 / 3, 1 / 3]])
    y_true = np.array([0])
    assert rank_probability_score(y_true, probs) == pytest.approx(5 / 18, rel=1e-5)


def test_when_two_matches_provided_then_rps_is_arithmetic_mean_of_per_match_values():
    """
    Match 1: p=[0.7, 0.2, 0.1], outcome=Win(0)
      CDF_pred=[0.7, 0.9]  CDF_obs=[1, 1]
      RPS_1 = 0.5*((0.3)^2 + (0.1)^2) = 0.5*(0.09+0.01) = 0.05

    Match 2: p=[0.2, 0.5, 0.3], outcome=Draw(1)
      CDF_pred=[0.2, 0.7]  CDF_obs=[0, 1]
      RPS_2 = 0.5*((0.2)^2 + (0.3)^2) = 0.5*(0.04+0.09) = 0.065

    Mean RPS = (0.05 + 0.065) / 2 = 0.0575
    """
    probs = np.array([[0.7, 0.2, 0.1], [0.2, 0.5, 0.3]])
    y_true = np.array([0, 1])
    assert rank_probability_score(y_true, probs) == pytest.approx(0.0575, rel=1e-5)


# ---------------------------------------------------------------------------
# multiclass_log_loss — example tests
# ---------------------------------------------------------------------------


def test_when_single_win_predicted_at_0_7_then_log_loss_equals_negative_log_0_7():
    """
    p=[0.7, 0.2, 0.1], outcome=Win(0).
    LL = -log(0.7)  ≈  0.35667
    """
    probs = np.array([[0.7, 0.2, 0.1]])
    y_true = np.array([0])
    expected = -math.log(0.7)
    assert multiclass_log_loss(y_true, probs) == pytest.approx(expected, rel=1e-5)


def test_when_two_matches_then_log_loss_is_mean_of_individual_losses():
    """
    Match 1: p=[0.7, 0.2, 0.1], outcome=Win   → LL = -log(0.7)  ≈ 0.35667
    Match 2: p=[0.2, 0.5, 0.3], outcome=Draw  → LL = -log(0.5)  ≈ 0.69315
    Mean LL = (-log(0.7) + -log(0.5)) / 2  ≈ 0.52491
    """
    probs = np.array([[0.7, 0.2, 0.1], [0.2, 0.5, 0.3]])
    y_true = np.array([0, 1])
    expected = (-math.log(0.7) + -math.log(0.5)) / 2
    assert multiclass_log_loss(y_true, probs) == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# brier_score — example tests
# ---------------------------------------------------------------------------


def test_when_perfect_one_hot_prediction_then_brier_score_is_zero():
    """p=[1,0,0], outcome=Win(0) → all squared errors are 0 → BS = 0."""
    probs = np.array([[1.0, 0.0, 0.0]])
    y_true = np.array([0])
    assert brier_score(y_true, probs) == pytest.approx(0.0)


def test_when_single_match_then_brier_score_equals_sum_of_squared_errors():
    """
    p=[0.7, 0.2, 0.1], outcome=Win → one-hot=[1, 0, 0].
    BS = (0.7-1)^2 + (0.2-0)^2 + (0.1-0)^2
       = 0.09 + 0.04 + 0.01
       = 0.14
    """
    probs = np.array([[0.7, 0.2, 0.1]])
    y_true = np.array([0])
    assert brier_score(y_true, probs) == pytest.approx(0.14, rel=1e-5)


def test_when_two_matches_then_brier_score_is_mean_of_per_match_values():
    """
    Match 1: p=[0.7, 0.2, 0.1], outcome=Win   → BS = 0.14  (see test above)
    Match 2: p=[0.2, 0.5, 0.3], outcome=Draw  → one-hot=[0, 1, 0]
      BS = (0.2-0)^2 + (0.5-1)^2 + (0.3-0)^2 = 0.04 + 0.25 + 0.09 = 0.38
    Mean BS = (0.14 + 0.38) / 2 = 0.26
    """
    probs = np.array([[0.7, 0.2, 0.1], [0.2, 0.5, 0.3]])
    y_true = np.array([0, 1])
    assert brier_score(y_true, probs) == pytest.approx(0.26, rel=1e-5)


# ---------------------------------------------------------------------------
# Property-based tests (Hypothesis)
# Derived from the invariants implied by the metric definitions:
#   - RPS ∈ [0, 1]  (bounded ordered divergence)
#   - RPS = 0 iff predicted distribution = observed distribution (perfect score)
#   - Brier ≥ 0     (sum of squares is non-negative)
#   - log-loss ≥ 0  (cross-entropy is non-negative for p ∈ (0,1])
# ---------------------------------------------------------------------------


@st.composite
def _simplex_and_outcome(draw):
    """
    Strategy: draw a valid 3-class probability vector (each entry ≥ 0.01,
    normalised to sum to 1) and a random outcome label in {0, 1, 2}.
    """
    raw = draw(
        st.lists(
            st.floats(
                min_value=0.01,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=3,
            max_size=3,
        )
    )
    total = sum(raw)
    probs = [v / total for v in raw]
    outcome = draw(st.integers(min_value=0, max_value=2))
    return np.array([probs]), np.array([outcome])


@given(_simplex_and_outcome())  # type: ignore[call-arg]
def test_when_valid_probability_vector_then_rps_lies_in_unit_interval(args):
    """RPS ∈ [0, 1] for any valid 3-class probability simplex and any outcome."""
    probs, y_true = args
    result = rank_probability_score(y_true, probs)
    assert 0.0 - 1e-9 <= result <= 1.0 + 1e-9


@given(st.integers(min_value=0, max_value=2))
def test_when_predicted_distribution_matches_outcome_then_rps_is_zero(outcome):
    """
    One-hot prediction that exactly matches the true outcome → RPS = 0,
    regardless of which class (Win / Draw / Loss) occurred.
    """
    one_hot = [0.0, 0.0, 0.0]
    one_hot[outcome] = 1.0
    probs = np.array([one_hot])
    y_true = np.array([outcome])
    assert rank_probability_score(y_true, probs) == pytest.approx(0.0, abs=1e-9)


@given(_simplex_and_outcome())  # type: ignore[call-arg]
def test_when_valid_probability_vector_then_brier_score_is_nonnegative(args):
    """Brier score ≥ 0 for any valid probability vector and any outcome."""
    probs, y_true = args
    assert brier_score(y_true, probs) >= -1e-9


@given(_simplex_and_outcome())  # type: ignore[call-arg]
def test_when_valid_probability_vector_then_log_loss_is_nonnegative(args):
    """log-loss ≥ 0 for any valid (strictly positive) probability vector and any outcome."""
    probs, y_true = args
    assert multiclass_log_loss(y_true, probs) >= -1e-9


# ---------------------------------------------------------------------------
# Elo-prior-weight tuning backtest (Dixon-Coles + Elo path)
# ---------------------------------------------------------------------------

import pandas as pd  # noqa: E402

from worldcup_playoff.config import AppConfig  # noqa: E402
from worldcup_playoff.models.evaluation import (  # noqa: E402
    _outcome,
    _wdl_probs,
    backtest_elo_prior_weight,
)
from worldcup_playoff.simulation.poisson import TeamAbilities  # noqa: E402


class TestOutcome:
    def test_home_win_is_class_zero(self):
        assert _outcome(2, 1) == 0

    def test_draw_is_class_one(self):
        assert _outcome(1, 1) == 1

    def test_away_win_is_class_two(self):
        assert _outcome(0, 3) == 2


class TestWdlProbs:
    def _abilities(self) -> TeamAbilities:
        return TeamAbilities(
            attack={"Strong": 0.8, "Weak": -0.8},
            defence={"Strong": 0.6, "Weak": -0.6},
            home_adv=0.25,
            rho=-0.1,
            intercept=0.1,
        )

    def test_probs_sum_to_one(self):
        p = _wdl_probs(self._abilities(), "Strong", "Weak", max_goals=10)
        assert math.isclose(sum(p), 1.0, abs_tol=1e-9)

    def test_stronger_home_team_favoured(self):
        p = _wdl_probs(self._abilities(), "Strong", "Weak", max_goals=10)
        assert p[0] > p[2]  # P(home win) > P(away win)


def _synthetic_results(seed: int = 0) -> pd.DataFrame:
    """Deterministic martj42-schema frame: tiered teams 2010-2022 + a 2022 WC block."""
    rng = np.random.default_rng(seed)
    strength = {"A": 2.2, "B": 1.6, "C": 1.2, "D": 0.8, "E": 0.5, "F": 0.3}
    teams = list(strength)
    rows = []
    for year in range(2010, 2022):
        for _ in range(40):
            h, a = rng.choice(teams, size=2, replace=False)
            rows.append({
                "DATE": f"{year}-03-15", "HOME_TEAM": h, "AWAY_TEAM": a,
                "HOME_GOALS": int(rng.poisson(strength[h])),
                "AWAY_GOALS": int(rng.poisson(strength[a])),
                "TOURNAMENT": "Friendly", "NEUTRAL": False,
            })
    for h, a in [("A", "F"), ("B", "E"), ("C", "D"), ("A", "B")]:
        rows.append({
            "DATE": "2022-06-20", "HOME_TEAM": h, "AWAY_TEAM": a,
            "HOME_GOALS": int(rng.poisson(strength[h])),
            "AWAY_GOALS": int(rng.poisson(strength[a])),
            "TOURNAMENT": "FIFA World Cup", "NEUTRAL": True,
        })
    return pd.DataFrame(rows)


class TestBacktestEloPriorWeight:
    def test_returns_weight_indexed_metrics(self):
        cfg = AppConfig()
        table = backtest_elo_prior_weight(
            _synthetic_results(), cfg, weights=(0.0, 0.5, 1.0), years=[2022]
        )
        assert list(table.index) == [0.0, 0.5, 1.0]
        assert {"rps", "log_loss", "brier"}.issubset(table.columns)
        assert table["rps"].between(0.0, 1.0).all()
        assert np.isfinite(table[["rps", "log_loss", "brier"]].to_numpy()).all()

    def test_empty_when_no_matching_world_cup(self):
        cfg = AppConfig()
        table = backtest_elo_prior_weight(
            _synthetic_results(), cfg, weights=(0.0, 1.0), years=[1990]
        )
        assert table.empty
