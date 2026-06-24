"""
Tests for Issue #6 — Dixon-Coles bivariate-Poisson estimator.

Derived exclusively from acceptance criteria; no implementation source was read.
Module expected at: worldcup_playoff.simulation.poisson
Config expected at: worldcup_playoff.config (PoissonConfig, AppConfig)

Assumption for score_matrix: score_matrix(lambda_home, lambda_away, rho, max_goals) -> np.ndarray
Assumption for decay_weight: decay_weight(age_days: float, half_life_days: float) -> float
Assumption for PoissonConfig: has fields half_life_days, max_goals, optimizer_maxiter, random_seed
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings, strategies as st
from pydantic import ValidationError

from worldcup_playoff.simulation.poisson import (
    DixonColesEstimator,
    ScorelineSampler,
    TeamAbilities,
    decay_weight,
    dixon_coles_tau,
    lambdas,
    score_matrix,
)
from worldcup_playoff.config import AppConfig, PoissonConfig


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

_TEAMS = ("Alpha", "Beta", "Gamma", "Delta")


def _played_df(teams=_TEAMS, n_per_pair: int = 6, seed: int = 42) -> pd.DataFrame:
    """Minimal played-match DataFrame: DATE, HOME_TEAM, AWAY_TEAM, HOME_GOALS, AWAY_GOALS."""
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2021-01-01")
    rows = []
    for i, home in enumerate(teams):
        for j, away in enumerate(teams):
            if home == away:
                continue
            for k in range(n_per_pair):
                rows.append(
                    {
                        "DATE": base + pd.Timedelta(days=k * 10 + i * 2 + j),
                        "HOME_TEAM": home,
                        "AWAY_TEAM": away,
                        "HOME_GOALS": int(rng.poisson(1.5)),
                        "AWAY_GOALS": int(rng.poisson(1.0)),
                    }
                )
    return pd.DataFrame(rows)


def _played_df_with_na() -> pd.DataFrame:
    """Played matches mixed with unplayed (NA goals) rows."""
    played = _played_df()
    na_rows = pd.DataFrame(
        {
            "DATE": [pd.Timestamp("2025-06-01"), pd.Timestamp("2025-06-15")],
            "HOME_TEAM": ["Alpha", "Beta"],
            "AWAY_TEAM": ["Beta", "Gamma"],
            "HOME_GOALS": pd.array([pd.NA, pd.NA], dtype="Int64"),
            "AWAY_GOALS": pd.array([pd.NA, pd.NA], dtype="Int64"),
        }
    )
    return pd.concat([played, na_rows], ignore_index=True)


def _strong_vs_weak_df(seed: int = 7) -> pd.DataFrame:
    """Strong dominates Weak across many matches; Mid is filler to complete the grid."""
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2019-01-01")
    rows = []
    for i in range(40):
        rows += [
            {
                "DATE": base + pd.Timedelta(days=i * 14),
                "HOME_TEAM": "Strong",
                "AWAY_TEAM": "Weak",
                "HOME_GOALS": int(rng.integers(3, 6)),
                "AWAY_GOALS": int(rng.integers(0, 2)),
            },
            {
                "DATE": base + pd.Timedelta(days=i * 14 + 7),
                "HOME_TEAM": "Weak",
                "AWAY_TEAM": "Strong",
                "HOME_GOALS": int(rng.integers(0, 2)),
                "AWAY_GOALS": int(rng.integers(3, 6)),
            },
            {
                "DATE": base + pd.Timedelta(days=i * 14 + 3),
                "HOME_TEAM": "Mid",
                "AWAY_TEAM": "Weak",
                "HOME_GOALS": int(rng.integers(1, 3)),
                "AWAY_GOALS": int(rng.integers(0, 2)),
            },
            {
                "DATE": base + pd.Timedelta(days=i * 14 + 10),
                "HOME_TEAM": "Strong",
                "AWAY_TEAM": "Mid",
                "HOME_GOALS": int(rng.integers(2, 5)),
                "AWAY_GOALS": int(rng.integers(0, 2)),
            },
        ]
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 1. dixon_coles_tau
#    Criterion: exact correction for (0,0),(1,0),(0,1),(1,1); 1.0 otherwise;
#    ρ=0 ⇒ all τ==1.0; τ>0 for valid small ρ with default λ.
# ─────────────────────────────────────────────────────────────────────────────


class TestDixonColesTau:
    LH, LA, RHO = 1.5, 1.0, 0.1

    def test_when_score_is_0_0_then_correct_tau_is_returned(self):
        expected = 1.0 - self.LH * self.LA * self.RHO
        assert dixon_coles_tau(0, 0, self.LH, self.LA, self.RHO) == pytest.approx(expected)

    def test_when_score_is_1_0_then_correct_tau_is_returned(self):
        # τ(1,0) = 1 + λ_away * ρ  (Dixon-Coles 1997, eq. 2)
        expected = 1.0 + self.LA * self.RHO
        assert dixon_coles_tau(1, 0, self.LH, self.LA, self.RHO) == pytest.approx(expected)

    def test_when_score_is_0_1_then_correct_tau_is_returned(self):
        # τ(0,1) = 1 + λ_home * ρ
        expected = 1.0 + self.LH * self.RHO
        assert dixon_coles_tau(0, 1, self.LH, self.LA, self.RHO) == pytest.approx(expected)

    def test_when_score_is_1_1_then_correct_tau_is_returned(self):
        # τ(1,1) = 1 - ρ
        expected = 1.0 - self.RHO
        assert dixon_coles_tau(1, 1, self.LH, self.LA, self.RHO) == pytest.approx(expected)

    @pytest.mark.parametrize("h, a", [(2, 0), (0, 2), (2, 2), (3, 1), (1, 3), (10, 10)])
    def test_when_score_is_at_least_two_then_tau_is_one(self, h, a):
        assert dixon_coles_tau(h, a, self.LH, self.LA, self.RHO) == pytest.approx(1.0)

    @pytest.mark.parametrize("h, a", [(0, 0), (1, 0), (0, 1), (1, 1)])
    def test_when_rho_is_zero_then_tau_is_one_for_each_low_score_cell(self, h, a):
        assert dixon_coles_tau(h, a, self.LH, self.LA, rho=0.0) == pytest.approx(1.0)

    @pytest.mark.parametrize("h, a", [(0, 0), (1, 0), (0, 1), (1, 1)])
    def test_when_rho_is_small_positive_and_lambdas_typical_then_tau_is_positive(self, h, a):
        assert dixon_coles_tau(h, a, self.LH, self.LA, rho=0.1) > 0.0

    @given(
        st.integers(min_value=0, max_value=1),
        st.integers(min_value=0, max_value=1),
    )
    def test_when_rho_zero_then_tau_is_one_for_any_low_score_cell(self, h, a):
        """Property: ρ=0 ⇒ τ==1.0 for all (h, a) in {0, 1}²."""
        assert dixon_coles_tau(h, a, 1.5, 1.0, 0.0) == pytest.approx(1.0)

    @given(
        st.integers(min_value=0, max_value=20),
        st.integers(min_value=0, max_value=20),
        st.floats(min_value=-0.99, max_value=0.99, allow_nan=False, allow_infinity=False),
    )
    def test_when_high_score_then_tau_is_one_regardless_of_rho(self, h, a, rho):
        """Property: h≥2 or a≥2 ⇒ τ==1.0 for any ρ."""
        assume(h >= 2 or a >= 2)
        assert dixon_coles_tau(h, a, 1.5, 1.0, rho) == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 2. DixonColesEstimator.fit — return type and NA handling
#    Criterion: returns TeamAbilities(attack, defence, home_adv, rho, intercept);
#    NA-goal rows dropped by _prepare; fit never raises on their presence.
# ─────────────────────────────────────────────────────────────────────────────


class TestDixonColesEstimatorFit:
    def test_when_fit_on_played_data_then_returns_team_abilities(self):
        result = DixonColesEstimator().fit(_played_df())
        assert isinstance(result, TeamAbilities)

    def test_when_fit_then_team_abilities_has_all_required_fields(self):
        abilities = DixonColesEstimator().fit(_played_df())
        for field in ("attack", "defence", "home_adv", "rho", "intercept"):
            assert hasattr(abilities, field), f"TeamAbilities missing field: {field}"

    def test_when_na_rows_present_then_fit_does_not_raise(self):
        DixonColesEstimator().fit(_played_df_with_na())

    def test_when_na_rows_present_then_fitted_teams_equal_played_only_teams(self):
        """NA rows are silently dropped; only played-match teams appear in abilities."""
        ab_played = DixonColesEstimator().fit(_played_df())
        ab_mixed = DixonColesEstimator().fit(_played_df_with_na())
        assert set(ab_played.attack.keys()) == set(ab_mixed.attack.keys())


# ─────────────────────────────────────────────────────────────────────────────
# 3. Time-decay weights
#    Criterion: reference date → 1.0; half_life_days old → 0.5;
#    2×half_life old → 0.25 (all pytest.approx).
#    Assumption: module exposes decay_weight(age_days, half_life_days) -> float.
# ─────────────────────────────────────────────────────────────────────────────


class TestTimeDecayWeight:
    HALF_LIFE = 180.0

    def test_when_age_is_zero_days_then_weight_is_one(self):
        assert decay_weight(0.0, self.HALF_LIFE) == pytest.approx(1.0)

    def test_when_age_is_one_half_life_then_weight_is_half(self):
        assert decay_weight(self.HALF_LIFE, self.HALF_LIFE) == pytest.approx(0.5)

    def test_when_age_is_two_half_lives_then_weight_is_quarter(self):
        assert decay_weight(2.0 * self.HALF_LIFE, self.HALF_LIFE) == pytest.approx(0.25)

    @given(
        st.floats(min_value=0.0, max_value=3650.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=1.0, max_value=730.0, allow_nan=False, allow_infinity=False),
    )
    def test_when_age_grows_by_one_day_then_weight_is_non_increasing(self, age, half_life):
        """Property: exponential decay is monotonically non-increasing in age."""
        w_now = decay_weight(age, half_life)
        w_older = decay_weight(age + 1.0, half_life)
        assert w_older <= w_now + 1e-12


# ─────────────────────────────────────────────────────────────────────────────
# 4. Determinism
#    Criterion: fitting twice on identical data yields identical abilities (tight approx).
# ─────────────────────────────────────────────────────────────────────────────


class TestFitDeterminism:
    def test_when_fit_twice_on_same_data_then_abilities_are_identical(self):
        df = _played_df()
        ab1 = DixonColesEstimator().fit(df)
        ab2 = DixonColesEstimator().fit(df)
        for team in ab1.attack:
            assert ab1.attack[team] == pytest.approx(ab2.attack[team], abs=1e-9)
        for team in ab1.defence:
            assert ab1.defence[team] == pytest.approx(ab2.defence[team], abs=1e-9)
        assert ab1.home_adv == pytest.approx(ab2.home_adv, abs=1e-9)
        assert ab1.rho == pytest.approx(ab2.rho, abs=1e-9)
        assert ab1.intercept == pytest.approx(ab2.intercept, abs=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Mean-zero attack normalization
#    Criterion: fitted attack values satisfy mean ≈ 0 (identifiability constraint).
# ─────────────────────────────────────────────────────────────────────────────


class TestMeanZeroNormalization:
    def test_when_fitted_then_mean_of_attack_values_is_approximately_zero(self):
        abilities = DixonColesEstimator().fit(_played_df())
        assert np.mean(list(abilities.attack.values())) == pytest.approx(0.0, abs=1e-6)

    def test_when_fitted_then_mean_of_defence_values_is_approximately_zero(self):
        abilities = DixonColesEstimator().fit(_played_df())
        assert np.mean(list(abilities.defence.values())) == pytest.approx(0.0, abs=1e-6)

    def test_when_normalization_applied_then_fitted_lambdas_are_unchanged(self):
        """Normalization must compensate intercept so all fitted λ remain identical."""
        from worldcup_playoff.simulation.poisson import _normalize_params
        import numpy as np

        n = 3
        # Craft params with non-zero attack and defence means.
        atk = np.array([0.4, -0.1, 0.3])
        dfn = np.array([0.2, 0.5, -0.1])
        home_adv, rho, intercept = 0.25, -0.1, 0.1
        params = np.concatenate([atk, dfn, [home_adv, rho, intercept]])
        normed = _normalize_params(params, n)

        # λ_home for each ordered pair must be invariant.
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                lh_orig = np.exp(params[2 * n + 2] + params[i] - params[n + j] + home_adv)
                lh_norm = np.exp(normed[2 * n + 2] + normed[i] - normed[n + j] + home_adv)
                assert lh_orig == pytest.approx(lh_norm, rel=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Recovery sanity
#    Criterion: seeded synthetic data where a clearly stronger team gets a higher
#    fitted attack than a weak team.
# ─────────────────────────────────────────────────────────────────────────────


class TestRecoverySanity:
    def test_when_strong_team_consistently_outscores_weak_team_then_higher_attack_is_fitted(self):
        abilities = DixonColesEstimator().fit(_strong_vs_weak_df())
        assert abilities.attack["Strong"] > abilities.attack["Weak"]


# ─────────────────────────────────────────────────────────────────────────────
# 7. ScorelineSampler.sample
#    Criterion: returns int array (size, 2) with goals in [0, max_goals];
#    reproducible with same seed; divergent across seeds;
#    default seed comes from PoissonConfig.
# ─────────────────────────────────────────────────────────────────────────────


class TestScorelineSampler:
    @pytest.fixture
    def abilities(self):
        return DixonColesEstimator().fit(_played_df())

    @pytest.fixture
    def sampler(self, abilities):
        return ScorelineSampler(abilities=abilities, config=PoissonConfig())

    def test_when_sample_called_then_result_shape_is_size_by_two(self, sampler):
        result = sampler.sample("Alpha", "Beta", size=100, random_state=0)
        assert result.shape == (100, 2)

    def test_when_sample_called_then_dtype_is_integer(self, sampler):
        result = sampler.sample("Alpha", "Beta", size=50, random_state=1)
        assert np.issubdtype(result.dtype, np.integer)

    def test_when_sample_called_then_goals_are_within_zero_and_max_goals(self, sampler):
        max_g = PoissonConfig().max_goals
        result = sampler.sample("Alpha", "Beta", size=500, random_state=2)
        assert int(result.min()) >= 0
        assert int(result.max()) <= max_g

    def test_when_same_random_state_used_then_output_is_reproducible(self, sampler):
        r1 = sampler.sample("Alpha", "Beta", size=50, random_state=42)
        r2 = sampler.sample("Alpha", "Beta", size=50, random_state=42)
        np.testing.assert_array_equal(r1, r2)

    def test_when_different_random_states_used_then_outputs_diverge(self, sampler):
        r1 = sampler.sample("Alpha", "Beta", size=300, random_state=0)
        r2 = sampler.sample("Alpha", "Beta", size=300, random_state=9999)
        assert not np.array_equal(r1, r2)

    def test_when_no_random_state_given_then_default_seed_comes_from_poisson_config(self, sampler):
        """Assumption: PoissonConfig exposes a `random_seed` field used as the default."""
        cfg = PoissonConfig()
        r_default = sampler.sample("Alpha", "Beta", size=50)
        r_explicit = sampler.sample("Alpha", "Beta", size=50, random_state=cfg.random_seed)
        np.testing.assert_array_equal(r_default, r_explicit)


# ─────────────────────────────────────────────────────────────────────────────
# 8. lambdas — neutral venue drops home advantage
#    Criterion: lambdas(abilities, home, away, neutral=True) drops home_adv ⇒
#    strictly lower λ_home than the non-neutral case.
# ─────────────────────────────────────────────────────────────────────────────


class TestLambdas:
    @pytest.fixture
    def abilities(self):
        return DixonColesEstimator().fit(_played_df())

    def test_when_neutral_true_then_lambda_home_is_strictly_lower_than_non_neutral(self, abilities):
        lh_non_neutral, _ = lambdas(abilities, "Alpha", "Beta", neutral=False)
        lh_neutral, _ = lambdas(abilities, "Alpha", "Beta", neutral=True)
        assert lh_neutral < lh_non_neutral

    def test_when_neutral_flag_toggled_then_lambda_away_is_unchanged(self, abilities):
        _, la_non_neutral = lambdas(abilities, "Alpha", "Beta", neutral=False)
        _, la_neutral = lambdas(abilities, "Alpha", "Beta", neutral=True)
        assert la_neutral == pytest.approx(la_non_neutral)


# ─────────────────────────────────────────────────────────────────────────────
# 9. score_matrix — joint pmf grid
#    Criterion: sums to 1.0 (approx); equals product of marginals when ρ=0.
#    Assumption: score_matrix(lambda_home, lambda_away, rho, max_goals) -> ndarray.
# ─────────────────────────────────────────────────────────────────────────────


class TestScoreMatrix:
    def test_when_score_matrix_computed_then_probabilities_sum_to_one(self):
        m = score_matrix(1.5, 1.0, rho=0.1, max_goals=15)
        assert m.sum() == pytest.approx(1.0, abs=1e-5)

    def test_when_rho_is_zero_then_matrix_equals_product_of_row_and_col_marginals(self):
        m = score_matrix(1.5, 1.0, rho=0.0, max_goals=15)
        row_marginals = m.sum(axis=1)
        col_marginals = m.sum(axis=0)
        expected = np.outer(row_marginals, col_marginals)
        np.testing.assert_allclose(m, expected, atol=1e-9)

    @given(
        st.floats(min_value=0.1, max_value=4.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.1, max_value=4.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50)
    def test_when_valid_lambdas_any_value_then_score_matrix_sums_to_one(self, lh, la):
        """Property: score_matrix with ρ=0 sums to ≈1.0 for any λ in (0.1, 4.0]."""
        m = score_matrix(lh, la, rho=0.0, max_goals=20)
        assert m.sum() == pytest.approx(1.0, abs=1e-3)


# ─────────────────────────────────────────────────────────────────────────────
# 10. PoissonConfig validation
#     Criterion: defaults load; half_life_days≤0, max_goals<1, optimizer_maxiter<1
#     raise ValidationError; AppConfig().poisson is a PoissonConfig.
# ─────────────────────────────────────────────────────────────────────────────


class TestPoissonConfig:
    def test_when_no_arguments_given_then_poisson_config_loads(self):
        assert PoissonConfig() is not None

    def test_when_half_life_days_is_zero_then_validation_error_is_raised(self):
        with pytest.raises(ValidationError):
            PoissonConfig(half_life_days=0)

    def test_when_half_life_days_is_negative_then_validation_error_is_raised(self):
        with pytest.raises(ValidationError):
            PoissonConfig(half_life_days=-1)

    def test_when_max_goals_is_zero_then_validation_error_is_raised(self):
        with pytest.raises(ValidationError):
            PoissonConfig(max_goals=0)

    def test_when_max_goals_is_negative_then_validation_error_is_raised(self):
        with pytest.raises(ValidationError):
            PoissonConfig(max_goals=-1)

    def test_when_optimizer_maxiter_is_zero_then_validation_error_is_raised(self):
        with pytest.raises(ValidationError):
            PoissonConfig(optimizer_maxiter=0)

    def test_when_optimizer_maxiter_is_negative_then_validation_error_is_raised(self):
        with pytest.raises(ValidationError):
            PoissonConfig(optimizer_maxiter=-1)

    def test_when_app_config_created_then_poisson_attribute_is_poisson_config_instance(self):
        assert isinstance(AppConfig().poisson, PoissonConfig)
