"""
Source-blind example tests for Issue #33:
  feat: Dixon-Coles τ-corrected scoreline sampler (simulation/poisson.py)

Every test is derived solely from the acceptance criteria text and the
requirements document.  No implementation file was read.  All tests are
Red-phase: they must fail until the implementation satisfies each criterion.

Criteria under test (UNIT-verifiable only):
  AC-1  ScorelineSampler.__call__(home, away, rng) draws one (home_goals,
        away_goals) pair advancing the injected Generator; .sample(...) returns
        an (size, 2) int array.
  AC-2  score_matrix(lh, la, rho, max_goals) applies the τ correction and is
        normalized to sum 1.0.
  AC-3  Deterministic given a fixed seed / random_state; the neutral flag drops
        home_adv.
  AC-4  make_sampler(abilities, config) factory provided.
"""

from __future__ import annotations

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from worldcup_playoff.config import PoissonConfig
from worldcup_playoff.simulation.poisson import ScorelineSampler, make_sampler, score_matrix

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Minimal abilities dict: team_name → {attack: float, defence: float}
ABILITIES: dict[str, dict[str, float]] = {
    "TeamA": {"attack": 1.5, "defence": 0.8},
    "TeamB": {"attack": 1.2, "defence": 1.0},
}

# Config with clearly non-unity home advantage so the neutral flag is testable.
# Designed to have all attributes the Dixon-Coles module needs.
_CFG = PoissonConfig(home_adv_init=1.35, rho_init=-0.1, max_goals=10)


# ---------------------------------------------------------------------------
# AC-2 — score_matrix: τ correction + normalization
# ---------------------------------------------------------------------------


class TestScoreMatrix:
    def test_when_score_matrix_called_then_probabilities_sum_to_one(self):
        """score_matrix must be normalized so all cell values sum to exactly 1.0."""
        mat = score_matrix(lh=1.5, la=1.2, rho=-0.1, max_goals=10)
        assert abs(mat.sum() - 1.0) < 1e-6

    def test_when_score_matrix_called_then_shape_is_max_goals_plus_one_squared(self):
        """Return shape must be (max_goals+1, max_goals+1)."""
        max_goals = 8
        mat = score_matrix(lh=1.5, la=1.2, rho=-0.1, max_goals=max_goals)
        assert mat.shape == (max_goals + 1, max_goals + 1)

    def test_when_score_matrix_called_then_all_entries_are_non_negative(self):
        """Normalized probabilities must all be ≥ 0."""
        mat = score_matrix(lh=1.5, la=1.2, rho=-0.1, max_goals=10)
        assert (mat >= 0).all()

    def test_when_rho_nonzero_then_matrix_differs_from_rho_zero(self):
        """τ correction must change the score matrix whenever rho ≠ 0."""
        lh, la, max_goals = 1.5, 1.2, 10
        mat_zero = score_matrix(lh=lh, la=la, rho=0.0, max_goals=max_goals)
        mat_corr = score_matrix(lh=lh, la=la, rho=-0.1, max_goals=max_goals)
        assert not np.allclose(mat_zero, mat_corr), (
            "rho != 0 must produce a matrix different from the uncorrected (rho=0) case"
        )

    def test_when_negative_rho_then_zero_zero_inflated_relative_to_one_zero(self):
        """
        With negative rho, τ(0,0) = 1 − lh·la·rho > 1 and τ(1,0) = 1 + la·rho < 1.

        The ratio M[0,0] / M[1,0] is normalization-invariant and must be strictly
        larger under negative rho than under rho=0, confirming the τ correction fires.
        """
        lh, la, max_goals = 1.5, 1.2, 10
        mat_zero = score_matrix(lh=lh, la=la, rho=0.0, max_goals=max_goals)
        mat_corr = score_matrix(lh=lh, la=la, rho=-0.1, max_goals=max_goals)
        ratio_zero = mat_zero[0, 0] / mat_zero[1, 0]
        ratio_corr = mat_corr[0, 0] / mat_corr[1, 0]
        assert ratio_corr > ratio_zero, (
            "Negative rho must inflate M[0,0] and deflate M[1,0] (τ>1 and τ<1 respectively)"
        )

    def test_when_positive_rho_then_one_one_deflated_relative_to_rho_zero(self):
        """
        With positive rho, τ(1,1) = 1 − rho < 1, so M[1,1] must be smaller than
        with rho=0 (before normalization; the ratio M[1,1]/M[2,2] must shrink).
        """
        lh, la, max_goals = 1.5, 1.2, 10
        mat_zero = score_matrix(lh=lh, la=la, rho=0.0, max_goals=max_goals)
        mat_pos = score_matrix(lh=lh, la=la, rho=0.1, max_goals=max_goals)
        # M[2,2] is unaffected by τ in both matrices (i+j > 1 for i=2, j=2)
        ratio_zero = mat_zero[1, 1] / mat_zero[2, 2]
        ratio_pos = mat_pos[1, 1] / mat_pos[2, 2]
        assert ratio_pos < ratio_zero, (
            "Positive rho must deflate M[1,1] relative to higher-score cells (τ(1,1)=1-rho<1)"
        )


# ---------------------------------------------------------------------------
# AC-2 property: score_matrix invariants hold for all valid inputs
# ---------------------------------------------------------------------------


@given(
    lh=st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
    la=st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
    rho=st.floats(min_value=-0.4, max_value=0.4, allow_nan=False, allow_infinity=False),
    max_goals=st.integers(min_value=5, max_value=20),
)
@settings(max_examples=200)
def test_when_valid_inputs_given_then_score_matrix_sums_to_one(
    lh: float, la: float, rho: float, max_goals: int
) -> None:
    """score_matrix is normalized to 1.0 for any valid (lh, la, rho, max_goals)."""
    # τ(0,0) = 1 − lh·la·rho must be positive for the model to be well-defined.
    assume(1.0 - lh * la * rho > 0)
    mat = score_matrix(lh=lh, la=la, rho=rho, max_goals=max_goals)
    assert abs(mat.sum() - 1.0) < 1e-5


@given(
    lh=st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
    la=st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
    rho=st.floats(min_value=-0.4, max_value=0.4, allow_nan=False, allow_infinity=False),
    max_goals=st.integers(min_value=5, max_value=20),
)
@settings(max_examples=200)
def test_when_valid_inputs_given_then_score_matrix_entries_non_negative(
    lh: float, la: float, rho: float, max_goals: int
) -> None:
    """All score_matrix cells are non-negative for any valid inputs."""
    assume(1.0 - lh * la * rho > 0)
    mat = score_matrix(lh=lh, la=la, rho=rho, max_goals=max_goals)
    assert (mat >= 0).all()


# ---------------------------------------------------------------------------
# AC-1 — ScorelineSampler.__call__: single draw, Generator advancement
# ---------------------------------------------------------------------------


class TestScorelineSamplerCall:
    def test_when_call_invoked_then_single_scoreline_pair_is_returned(self):
        """__call__ must return a 2-element sequence (home_goals, away_goals)."""
        sampler = make_sampler(ABILITIES, _CFG)
        rng = np.random.default_rng(42)
        result = sampler("TeamA", "TeamB", rng)
        assert len(result) == 2

    def test_when_call_invoked_then_goals_are_non_negative_integers(self):
        """Both goal counts returned by __call__ must be non-negative integers."""
        sampler = make_sampler(ABILITIES, _CFG)
        rng = np.random.default_rng(42)
        home_goals, away_goals = sampler("TeamA", "TeamB", rng)
        assert isinstance(home_goals, (int, np.integer)), "home_goals must be an integer type"
        assert isinstance(away_goals, (int, np.integer)), "away_goals must be an integer type"
        assert home_goals >= 0
        assert away_goals >= 0

    def test_when_call_invoked_then_rng_state_is_advanced(self):
        """Calling __call__ must consume state from the injected Generator."""
        sampler = make_sampler(ABILITIES, _CFG)
        # Two generators with identical seeds — one gets advanced by __call__
        rng_advanced = np.random.default_rng(0)
        rng_reference = np.random.default_rng(0)
        sampler("TeamA", "TeamB", rng_advanced)
        # Draw several values; if the generator was advanced they will differ
        draw_advanced = [rng_advanced.integers(0, 100_000) for _ in range(5)]
        draw_reference = [rng_reference.integers(0, 100_000) for _ in range(5)]
        assert draw_advanced != draw_reference, (
            "__call__ must advance the injected Generator so subsequent draws differ"
        )


# ---------------------------------------------------------------------------
# AC-1 — ScorelineSampler.sample: (size, 2) int array
# ---------------------------------------------------------------------------


class TestScorelineSamplerSample:
    def test_when_sample_called_then_shape_is_size_by_2(self):
        """sample must return an array of shape (size, 2)."""
        sampler = make_sampler(ABILITIES, _CFG)
        result = sampler.sample("TeamA", "TeamB", size=100, random_state=0)
        assert result.shape == (100, 2)

    def test_when_sample_called_then_dtype_is_integer(self):
        """sample must return an integer-dtype array (goals are whole numbers)."""
        sampler = make_sampler(ABILITIES, _CFG)
        result = sampler.sample("TeamA", "TeamB", size=50, random_state=0)
        assert np.issubdtype(result.dtype, np.integer)

    def test_when_sample_called_then_all_values_are_non_negative(self):
        """No sampled goal count may be negative."""
        sampler = make_sampler(ABILITIES, _CFG)
        result = sampler.sample("TeamA", "TeamB", size=200, random_state=7)
        assert (result >= 0).all()

    def test_when_sample_size_one_then_shape_is_one_by_2(self):
        """sample(size=1) must still produce shape (1, 2), not a 1-D array."""
        sampler = make_sampler(ABILITIES, _CFG)
        result = sampler.sample("TeamA", "TeamB", size=1, random_state=3)
        assert result.shape == (1, 2)


# ---------------------------------------------------------------------------
# AC-1 property: sample shape and dtype hold for any positive size
# ---------------------------------------------------------------------------


@given(size=st.integers(min_value=1, max_value=500))
def test_when_sample_called_with_any_positive_size_then_shape_is_size_by_2(size: int) -> None:
    """sample must always return (size, 2) regardless of the requested count."""
    sampler = make_sampler(ABILITIES, _CFG)
    result = sampler.sample("TeamA", "TeamB", size=size, random_state=0)
    assert result.shape == (size, 2)


@given(size=st.integers(min_value=1, max_value=300))
def test_when_sample_called_with_any_positive_size_then_goals_are_non_negative_integers(
    size: int,
) -> None:
    """sample must return non-negative integer goal counts for any positive size."""
    sampler = make_sampler(ABILITIES, _CFG)
    result = sampler.sample("TeamA", "TeamB", size=size, random_state=42)
    assert np.issubdtype(result.dtype, np.integer)
    assert (result >= 0).all()


# ---------------------------------------------------------------------------
# AC-3 — Determinism and neutral flag
# ---------------------------------------------------------------------------


class TestDeterminismAndNeutral:
    def test_when_same_seed_passed_twice_to_call_then_results_are_identical(self):
        """Two __call__ invocations with the same seed must produce the same scoreline."""
        sampler = make_sampler(ABILITIES, _CFG)
        result_a = sampler("TeamA", "TeamB", np.random.default_rng(7))
        result_b = sampler("TeamA", "TeamB", np.random.default_rng(7))
        assert result_a == result_b

    def test_when_same_random_state_used_twice_then_sample_arrays_are_identical(self):
        """Two sample calls with the same random_state must produce bit-identical arrays."""
        sampler = make_sampler(ABILITIES, _CFG)
        result_a = sampler.sample("TeamA", "TeamB", size=100, random_state=99)
        result_b = sampler.sample("TeamA", "TeamB", size=100, random_state=99)
        np.testing.assert_array_equal(result_a, result_b)

    def test_when_neutral_true_then_sample_differs_from_home_advantage_sample(self):
        """
        neutral=True must drop home_adv, changing the scoring distribution.

        With _CFG.home_adv=1.35 (clearly > 1), the expected home-goal rate under
        neutral=False is strictly higher than under neutral=True.  Given a
        sufficiently large sample and fixed seed the means will diverge.

        Design choice: 'neutral' is a keyword argument to .sample(); if the
        implementation attaches it to __call__ instead, adjust accordingly.
        """
        sampler = make_sampler(ABILITIES, _CFG)
        n = 2_000
        with_adv = sampler.sample("TeamA", "TeamB", size=n, random_state=1, neutral=False)
        without_adv = sampler.sample("TeamA", "TeamB", size=n, random_state=1, neutral=True)
        assert not np.array_equal(with_adv, without_adv), (
            "neutral=True must drop home_adv; sample distributions must differ"
        )
        # Home-goal column: column 0.  With home advantage the mean must be higher.
        assert with_adv[:, 0].mean() > without_adv[:, 0].mean(), (
            "Home goals should be lower when neutral=True (home_adv removed)"
        )

    def test_when_neutral_true_then_call_result_differs_from_non_neutral(self):
        """
        Across many draws, __call__ with neutral=True must produce a lower average
        home-goal count than neutral=False (home_adv > 1 → neutral lowers lh).
        """
        sampler = make_sampler(ABILITIES, _CFG)
        n = 1_000
        home_goals_with = [
            sampler("TeamA", "TeamB", np.random.default_rng(i), neutral=False)[0] for i in range(n)
        ]
        home_goals_without = [
            sampler("TeamA", "TeamB", np.random.default_rng(i), neutral=True)[0] for i in range(n)
        ]
        assert np.mean(home_goals_with) > np.mean(home_goals_without), (
            "Average home goals must be higher with home advantage than on a neutral venue"
        )


# ---------------------------------------------------------------------------
# AC-4 — make_sampler factory
# ---------------------------------------------------------------------------


class TestMakeSampler:
    def test_when_make_sampler_called_then_sampler_is_callable(self):
        """make_sampler must return an object that is callable (__call__ defined)."""
        sampler = make_sampler(ABILITIES, _CFG)
        assert callable(sampler)

    def test_when_make_sampler_called_then_result_is_scoreline_sampler_instance(self):
        """make_sampler must return a ScorelineSampler instance."""
        sampler = make_sampler(ABILITIES, _CFG)
        assert isinstance(sampler, ScorelineSampler)

    def test_when_make_sampler_called_then_sampler_has_sample_method(self):
        """The returned sampler must expose a callable .sample attribute."""
        sampler = make_sampler(ABILITIES, _CFG)
        assert hasattr(sampler, "sample"), "ScorelineSampler must have a .sample method"
        assert callable(sampler.sample)

    def test_when_make_sampler_called_with_different_abilities_then_samplers_differ(self):
        """
        Two samplers built with different ability sets must produce different
        scoreline distributions (under identical seeds).
        """
        abilities_strong_home = {
            "TeamA": {"attack": 3.0, "defence": 0.5},
            "TeamB": {"attack": 0.5, "defence": 2.0},
        }
        abilities_weak_home = {
            "TeamA": {"attack": 0.5, "defence": 2.0},
            "TeamB": {"attack": 3.0, "defence": 0.5},
        }
        sampler_strong = make_sampler(abilities_strong_home, _CFG)
        sampler_weak = make_sampler(abilities_weak_home, _CFG)
        arr_strong = sampler_strong.sample("TeamA", "TeamB", size=500, random_state=0)
        arr_weak = sampler_weak.sample("TeamA", "TeamB", size=500, random_state=0)
        # Strong home-team attack → higher average home goals
        assert arr_strong[:, 0].mean() > arr_weak[:, 0].mean(), (
            "Sampler with stronger home-team attack must yield higher home-goal averages"
        )
