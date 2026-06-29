"""Tests for the squad-market-value prior."""

from __future__ import annotations

import math

from worldcup_playoff.data.squad_value import WC2026_SQUAD_VALUE_EUR_M
from worldcup_playoff.simulation.poisson import (
    TeamAbilities,
    blend_abilities_with_market_value,
)


def _abilities() -> TeamAbilities:
    # Fitted abilities rank B above A (B has the higher attack/defence). The
    # market value ranks them the *other* way, so the blend should pull A up and
    # B down. Non-zero spread is required — the prior is rescaled to the fitted
    # spread, so identical abilities would leave nothing to reorder.
    return TeamAbilities(
        attack={"A": -1.0, "B": 1.0},
        defence={"A": -1.0, "B": 1.0},
        home_adv=0.3,
        rho=-0.1,
        intercept=0.1,
    )


def test_squad_value_table_has_48_positive_entries():
    assert len(WC2026_SQUAD_VALUE_EUR_M) == 48
    assert all(v > 0 for v in WC2026_SQUAD_VALUE_EUR_M.values())


def test_zero_weight_leaves_abilities_unchanged():
    ab = _abilities()
    out = blend_abilities_with_market_value(ab, {"A": 1000.0, "B": 10.0}, 0.0)
    assert out.attack == ab.attack
    assert out.defence == ab.defence


def test_richer_squad_is_pulled_up_poorer_pulled_down():
    ab = _abilities()
    out = blend_abilities_with_market_value(ab, {"A": 1000.0, "B": 10.0}, 0.5)
    # A is the richer squad but the weaker fit → blend lifts it; B is pulled down.
    assert out.attack["A"] > ab.attack["A"]
    assert out.attack["B"] < ab.attack["B"]
    assert out.defence["A"] > ab.defence["A"]
    assert out.defence["B"] < ab.defence["B"]
    # Prior is symmetric around the shared mean (0 here).
    assert math.isclose(out.attack["A"], -out.attack["B"], abs_tol=1e-9)


def test_team_without_a_value_is_left_untouched():
    ab = _abilities()
    # Only one team has a value → fewer than 2 shared teams → unchanged.
    out = blend_abilities_with_market_value(ab, {"A": 1000.0}, 0.5)
    assert out.attack == ab.attack
