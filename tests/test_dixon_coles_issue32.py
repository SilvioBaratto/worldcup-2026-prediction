"""
Source-blind example tests for Issue #32.
Dixon-Coles bivariate-Poisson estimator — worldcup_playoff/simulation/poisson.py.

Tests derived exclusively from the acceptance criteria; no implementation was read.
Criteria tested (oracle classification drives which ones are authored):
  [UNIT] fit() returns TeamAbilities(attack, defence, home_adv, rho, intercept)
         for every team in played history.
  [UNIT] Exponential time-decay via half_life_days; rho τ-correction; neutral drops home_adv.
  [UNIT] lambdas(abilities, home, away, neutral) returns (λ_home, λ_away).
Skipped (oracle: NOT VERIFIABLE):
  - Attack abilities mean-zero normalization (no concrete runtime check inferable)
  - Deterministic no-key run (boilerplate / env constraint, not a unit assertion)
  - All-tests-pass / SOLID / TDD (suite gate + subjective quality prose)
"""

import numpy as np
import pandas as pd
from hypothesis import given, settings, strategies as st


# ---------------------------------------------------------------------------
# Deferred imports — tests collect even before implementation exists
# ---------------------------------------------------------------------------


def _estimator():
    from worldcup_playoff.simulation.poisson import DixonColesEstimator

    return DixonColesEstimator


def _lambdas():
    from worldcup_playoff.simulation.poisson import lambdas

    return lambdas


def _team_abilities_cls():
    from worldcup_playoff.simulation.poisson import TeamAbilities

    return TeamAbilities


# ---------------------------------------------------------------------------
# Fixtures built from acceptance criteria and the internal matches.csv schema
# (DATE, HOME_TEAM, AWAY_TEAM, HOME_GOALS, AWAY_GOALS, NEUTRAL)
# ---------------------------------------------------------------------------

_BASE = pd.Timestamp("2022-01-01")
_DAY = pd.Timedelta(days=30)


def _minimal_df() -> pd.DataFrame:
    """Ten-match round-robin over five teams; all non-neutral."""
    rows = [
        ("France", "Argentina", 2, 1, False),
        ("Brazil", "Spain", 1, 2, False),
        ("England", "France", 0, 1, False),
        ("Argentina", "Brazil", 3, 0, False),
        ("Spain", "England", 2, 2, False),
        ("France", "Brazil", 1, 0, False),
        ("Argentina", "Spain", 1, 1, False),
        ("Brazil", "England", 2, 1, False),
        ("Spain", "France", 0, 2, False),
        ("England", "Argentina", 1, 2, False),
    ]
    return pd.DataFrame(
        [
            {
                "DATE": _BASE + _DAY * i,
                "HOME_TEAM": h,
                "AWAY_TEAM": a,
                "HOME_GOALS": hg,
                "AWAY_GOALS": ag,
                "NEUTRAL": n,
            }
            for i, (h, a, hg, ag, n) in enumerate(rows)
        ]
    )


def _df_with_neutral() -> pd.DataFrame:
    """Same five teams; four matches flagged neutral=True to exercise that branch."""
    rows = [
        ("France", "Argentina", 2, 1, False),
        ("Brazil", "Spain", 1, 2, False),
        ("England", "France", 0, 1, False),
        ("France", "Brazil", 1, 0, True),
        ("Argentina", "Spain", 1, 1, True),
        ("Brazil", "England", 2, 1, False),
        ("Spain", "France", 0, 2, False),
        ("England", "Argentina", 1, 2, False),
        ("Argentina", "Brazil", 2, 1, True),
        ("Spain", "England", 1, 0, True),
    ]
    return pd.DataFrame(
        [
            {
                "DATE": _BASE + _DAY * i,
                "HOME_TEAM": h,
                "AWAY_TEAM": a,
                "HOME_GOALS": hg,
                "AWAY_GOALS": ag,
                "NEUTRAL": n,
            }
            for i, (h, a, hg, ag, n) in enumerate(rows)
        ]
    )


# ===========================================================================
# Criterion 1 — fit() returns TeamAbilities with all five fields and covers
#               every team that appeared in the training DataFrame
# ===========================================================================


def test_when_fit_is_called_then_result_has_attack_field():
    result = _estimator()().fit(_minimal_df())
    assert hasattr(result, "attack")


def test_when_fit_is_called_then_result_has_defence_field():
    result = _estimator()().fit(_minimal_df())
    assert hasattr(result, "defence")


def test_when_fit_is_called_then_result_has_home_adv_field():
    result = _estimator()().fit(_minimal_df())
    assert hasattr(result, "home_adv")


def test_when_fit_is_called_then_result_has_rho_field():
    result = _estimator()().fit(_minimal_df())
    assert hasattr(result, "rho")


def test_when_fit_is_called_then_result_has_intercept_field():
    result = _estimator()().fit(_minimal_df())
    assert hasattr(result, "intercept")


def test_when_fit_is_called_then_every_team_appears_in_attack():
    df = _minimal_df()
    result = _estimator()().fit(df)
    all_teams = set(df["HOME_TEAM"]) | set(df["AWAY_TEAM"])
    for team in all_teams:
        assert team in result.attack, f"{team!r} missing from TeamAbilities.attack"


def test_when_fit_is_called_then_every_team_appears_in_defence():
    df = _minimal_df()
    result = _estimator()().fit(df)
    all_teams = set(df["HOME_TEAM"]) | set(df["AWAY_TEAM"])
    for team in all_teams:
        assert team in result.defence, f"{team!r} missing from TeamAbilities.defence"


def test_when_fit_is_called_then_home_adv_is_a_finite_float():
    result = _estimator()().fit(_minimal_df())
    assert isinstance(result.home_adv, float)
    assert np.isfinite(result.home_adv)


def test_when_fit_is_called_then_rho_is_a_finite_float():
    result = _estimator()().fit(_minimal_df())
    assert isinstance(result.rho, float)
    assert np.isfinite(result.rho)


def test_when_fit_is_called_then_intercept_is_a_finite_float():
    result = _estimator()().fit(_minimal_df())
    assert isinstance(result.intercept, float)
    assert np.isfinite(result.intercept)


def test_when_fit_is_called_then_all_attack_values_are_finite():
    result = _estimator()().fit(_minimal_df())
    for team, val in result.attack.items():
        assert np.isfinite(val), f"attack[{team!r}] = {val!r} is not finite"


def test_when_fit_is_called_then_all_defence_values_are_finite():
    result = _estimator()().fit(_minimal_df())
    for team, val in result.defence.items():
        assert np.isfinite(val), f"defence[{team!r}] = {val!r} is not finite"


# ===========================================================================
# Criterion 2 — half_life_days exponential decay, rho τ-correction, and
#               neutral matches that drop home_adv from λ_home
# ===========================================================================


def test_when_half_life_days_is_given_then_estimator_accepts_it():
    """Estimator must expose a half_life_days constructor parameter."""
    _estimator()(half_life_days=365).fit(_minimal_df())  # must not raise


def test_when_half_life_days_differs_then_fitted_abilities_differ():
    """
    A 30-day half-life weights only the most recent matches; a 3650-day half-life
    weights all matches nearly equally.  At least one team's attack must change.
    """
    df = _minimal_df()
    short = _estimator()(half_life_days=30).fit(df)
    long_ = _estimator()(half_life_days=3650).fit(df)
    diffs = [abs(short.attack[t] - long_.attack[t]) for t in short.attack]
    assert any(d > 1e-6 for d in diffs), (
        "Abilities must differ for different half_life_days values "
        "(exponential time-decay is active)"
    )


def test_when_rho_is_fitted_then_it_is_not_exactly_zero():
    """
    rho == 0 collapses the model to independent Poisson (no Dixon-Coles τ-correction).
    The criterion requires the correction to be applied during fitting.
    """
    result = _estimator()().fit(_minimal_df())
    assert result.rho != 0.0, "rho must be non-zero; rho=0 means no τ-correction"


def test_when_neutral_is_true_then_lambda_home_differs_from_non_neutral():
    """
    Neutral matches drop home_adv from λ_home.
    When home_adv != 0 the resulting λ_home must differ between neutral and non-neutral.
    """
    df = _df_with_neutral()
    abilities = _estimator()().fit(df)
    lambdas_fn = _lambdas()
    lam_h_neutral, _ = lambdas_fn(abilities, "France", "Argentina", neutral=True)
    lam_h_home, _ = lambdas_fn(abilities, "France", "Argentina", neutral=False)
    if abs(abilities.home_adv) > 1e-9:
        assert lam_h_neutral != lam_h_home, (
            "neutral=True must change λ_home when home_adv != 0 "
            "(neutral matches drop the home-advantage term)"
        )


def test_when_neutral_is_true_then_lambda_away_is_unchanged():
    """
    home_adv is only applied to the home side; λ_away must be identical
    for neutral=True and neutral=False with the same team pair.
    """
    abilities = _estimator()().fit(_df_with_neutral())
    lambdas_fn = _lambdas()
    _, lam_away_neutral = lambdas_fn(abilities, "France", "Argentina", neutral=True)
    _, lam_away_home = lambdas_fn(abilities, "France", "Argentina", neutral=False)
    assert abs(lam_away_neutral - lam_away_home) < 1e-9, (
        "λ_away must be identical regardless of the neutral flag"
    )


def test_when_neutral_true_and_teams_are_swapped_then_lambdas_are_mirror_swapped():
    """
    For a neutral match (no home-advantage term) swapping the team slots must
    swap λ_home and λ_away exactly:
        λ_home(A,B,neutral) == λ_away(B,A,neutral)
        λ_away(A,B,neutral) == λ_home(B,A,neutral)
    This confirms home_adv is absent, not merely sign-flipped.
    """
    abilities = _estimator()().fit(_df_with_neutral())
    lambdas_fn = _lambdas()
    lam_h, lam_a = lambdas_fn(abilities, "France", "Brazil", neutral=True)
    lam_h_rev, lam_a_rev = lambdas_fn(abilities, "Brazil", "France", neutral=True)
    assert abs(lam_h - lam_a_rev) < 1e-9, (
        "λ_home(France,Brazil,neutral) must equal λ_away(Brazil,France,neutral)"
    )
    assert abs(lam_a - lam_h_rev) < 1e-9, (
        "λ_away(France,Brazil,neutral) must equal λ_home(Brazil,France,neutral)"
    )


# ===========================================================================
# Criterion 4 — lambdas(abilities, home, away, neutral) exposes (λ_home, λ_away)
# ===========================================================================


def test_when_lambdas_is_called_then_two_values_are_returned():
    abilities = _estimator()().fit(_minimal_df())
    result = _lambdas()(abilities, "France", "Argentina", neutral=False)
    assert len(result) == 2


def test_when_lambdas_is_called_then_lambda_home_is_positive():
    abilities = _estimator()().fit(_minimal_df())
    lam_home, _ = _lambdas()(abilities, "France", "Argentina", neutral=False)
    assert lam_home > 0


def test_when_lambdas_is_called_then_lambda_away_is_positive():
    abilities = _estimator()().fit(_minimal_df())
    _, lam_away = _lambdas()(abilities, "France", "Argentina", neutral=False)
    assert lam_away > 0


def test_when_lambdas_called_with_neutral_true_then_lambda_home_is_positive():
    abilities = _estimator()().fit(_minimal_df())
    lam_home, _ = _lambdas()(abilities, "France", "Argentina", neutral=True)
    assert lam_home > 0


def test_when_lambdas_called_with_neutral_true_then_lambda_away_is_positive():
    abilities = _estimator()().fit(_minimal_df())
    _, lam_away = _lambdas()(abilities, "France", "Argentina", neutral=True)
    assert lam_away > 0


def test_when_neutral_false_and_home_adv_positive_then_lambda_home_exceeds_neutral_case():
    """
    When home_adv > 0 the home team's λ must be higher in a home match than in a
    neutral match (same teams, same abilities).
    """
    abilities = _estimator()().fit(_minimal_df())
    lambdas_fn = _lambdas()
    lam_h_neutral, _ = lambdas_fn(abilities, "France", "Argentina", neutral=True)
    lam_h_home, _ = lambdas_fn(abilities, "France", "Argentina", neutral=False)
    if abilities.home_adv > 0:
        assert lam_h_home > lam_h_neutral, (
            "home_adv > 0 must raise λ_home above the neutral-match value"
        )
    elif abilities.home_adv < 0:
        assert lam_h_home < lam_h_neutral, (
            "home_adv < 0 must lower λ_home below the neutral-match value"
        )


# ===========================================================================
# Property-based tests (Hypothesis)
# ===========================================================================


@given(
    teams=st.lists(
        st.text(
            min_size=3,
            max_size=20,
            alphabet=st.characters(whitelist_categories=["Lu", "Ll"]),
        ),
        min_size=3,
        max_size=6,
        unique=True,
    )
)
@settings(max_examples=8, deadline=None)
def test_when_fit_called_with_any_valid_teams_then_every_team_appears_in_abilities(teams):
    """
    Invariant (criterion 1): every team that appears as HOME_TEAM or AWAY_TEAM in the
    input DataFrame must have an entry in both TeamAbilities.attack and .defence.
    Strategy: build a full round-robin so every team plays at least once.
    """
    rows = []
    for i, home in enumerate(teams):
        for j, away in enumerate(teams):
            if i == j:
                continue
            rows.append(
                {
                    "DATE": _BASE + _DAY * (i * len(teams) + j),
                    "HOME_TEAM": home,
                    "AWAY_TEAM": away,
                    "HOME_GOALS": 1,
                    "AWAY_GOALS": 0,
                    "NEUTRAL": False,
                }
            )
    if not rows:
        return
    df = pd.DataFrame(rows)
    abilities = _estimator()().fit(df)
    all_teams = set(df["HOME_TEAM"]) | set(df["AWAY_TEAM"])
    for team in all_teams:
        assert team in abilities.attack, f"{team!r} missing from attack"
        assert team in abilities.defence, f"{team!r} missing from defence"


@given(
    home_atk=st.floats(min_value=-3.0, max_value=3.0, allow_nan=False, allow_infinity=False),
    away_atk=st.floats(min_value=-3.0, max_value=3.0, allow_nan=False, allow_infinity=False),
    home_def=st.floats(min_value=-3.0, max_value=3.0, allow_nan=False, allow_infinity=False),
    away_def=st.floats(min_value=-3.0, max_value=3.0, allow_nan=False, allow_infinity=False),
    home_adv=st.floats(min_value=-1.5, max_value=1.5, allow_nan=False, allow_infinity=False),
    intercept=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
    rho=st.floats(min_value=-0.5, max_value=0.0, allow_nan=False, allow_infinity=False),
    neutral=st.booleans(),
)
@settings(max_examples=50, deadline=None)
def test_when_lambdas_called_with_valid_abilities_then_both_rates_are_positive(
    home_atk,
    away_atk,
    home_def,
    away_def,
    home_adv,
    intercept,
    rho,
    neutral,
):
    """
    Invariant (criterion 4): lambdas must always return strictly positive Poisson rates.
    A rate ≤ 0 is physically undefined for a Poisson distribution and would break
    downstream scoreline sampling.  TeamAbilities is constructed directly (per the
    constructor signature stated in the criterion) to exercise lambdas in isolation.
    """
    TeamAbilities = _team_abilities_cls()
    abilities = TeamAbilities(
        attack={"A": home_atk, "B": away_atk},
        defence={"A": home_def, "B": away_def},
        home_adv=home_adv,
        rho=rho,
        intercept=intercept,
    )
    lam_home, lam_away = _lambdas()(abilities, "A", "B", neutral=neutral)
    assert lam_home > 0, f"λ_home must be positive; got {lam_home!r}"
    assert lam_away > 0, f"λ_away must be positive; got {lam_away!r}"
