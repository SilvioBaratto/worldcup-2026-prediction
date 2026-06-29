"""Tests for the skill-weighted penalty shootout."""

from __future__ import annotations

import numpy as np

from worldcup_playoff.simulation.knockout import _penalty_flip, _shootout_winner
from worldcup_playoff.simulation.poisson import TeamAbilities


def _abilities() -> TeamAbilities:
    return TeamAbilities(
        attack={"Strong": 1.0, "Weak": -1.0},
        defence={"Strong": 1.0, "Weak": -1.0},
        home_adv=0.0, rho=-0.1, intercept=0.0,
    )


def test_shootout_is_a_coin_flip_when_skill_is_zero():
    # Matches the seeded coin flip exactly for every seed.
    rng = np.random.default_rng(0)
    for seed in range(50):
        assert _shootout_winner(
            "Strong", "Weak", seed, _abilities(), None, rng, penalty_skill=0.0
        ) == _penalty_flip("Strong", "Weak", seed)


def test_shootout_is_a_coin_flip_without_abilities():
    rng = np.random.default_rng(0)
    out = _shootout_winner("Strong", "Weak", 7, None, None, rng, penalty_skill=1.0)
    assert out == _penalty_flip("Strong", "Weak", 7)


def test_shootout_favours_stronger_team_but_stays_capped():
    ab = _abilities()
    rng = np.random.default_rng(0)
    n = 4000
    wins = sum(
        _shootout_winner("Strong", "Weak", i, ab, None, rng, penalty_skill=1.0) == "Strong"
        for i in range(n)
    )
    frac = wins / n
    # Stronger team favoured, but the edge is capped near 0.5 (max tilt 0.15).
    assert 0.5 < frac <= 0.66
