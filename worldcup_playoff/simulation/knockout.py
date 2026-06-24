"""Single-elimination knockout simulator for WC2026.

Resolves the R32 bracket from group standings, then folds the tree
(R32 → R16 → QF → SF → Final) via Monte Carlo sampling:
  regulation → extra time (λ × extra_time_factor) → penalty coin-flip.

All knockout matches are treated as neutral-venue fixtures.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from worldcup_playoff.config import PoissonConfig, SimulationConfig
from worldcup_playoff.data.wc2026_bracket import R32_SLOTS, resolve_r32
from worldcup_playoff.simulation.poisson import (
    TeamAbilities,
    lambdas,
    score_matrix,
)
from worldcup_playoff.simulation.tournament import RoundResult

logger = logging.getLogger(__name__)

# Re-export for consumers who import from this module.
__all__ = [
    "KnockoutRound",
    "KnockoutSimulator",
    "R32_SLOTS",
    "resolve_r32",
    "resolve_tie",
    "simulate",
]

# Type alias for a scoreline sampler callable.
_Sampler = Callable[[str, str], tuple[int, int]]


# ---------------------------------------------------------------------------
# Pure helpers (module-level, ≤10 lines each)
# ---------------------------------------------------------------------------


def _draw_from_matrix(mat: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
    """Draw one (home_goals, away_goals) pair from a normalised score-matrix pmf."""
    g = mat.shape[0]
    idx = int(rng.choice(g * g, p=mat.ravel()))
    return idx // g, idx % g


def _penalty_flip(home: str, away: str, seed: int) -> str:
    """Return the penalty winner via a seeded coin-flip (0 → home, 1 → away)."""
    rng = np.random.default_rng(seed)
    return home if rng.integers(2) == 0 else away


def _et_goals(
    home: str,
    away: str,
    sampler: _Sampler,
    extra_time_factor: float,
    abilities: TeamAbilities | None,
    poisson_config: PoissonConfig | None,
    rng: np.random.Generator | None,
) -> tuple[int, int]:
    """Extra-time goals: scaled Poisson when abilities+rng available, else stub sampler."""
    if abilities is None or rng is None:
        return sampler(home, away)
    cfg = poisson_config or PoissonConfig()
    lh, la = lambdas(abilities, home, away, neutral=True)
    mat = score_matrix(lh * extra_time_factor, la * extra_time_factor,
                       rho=abilities.rho, max_goals=cfg.max_goals)
    return _draw_from_matrix(mat, rng)


def _record_round(
    rounds: dict[int, RoundResult],
    round_num: int,
    all_teams: list[str],
    winners: list[str],
) -> None:
    """Initialise (first call) and increment advancement counts for *round_num*."""
    if round_num not in rounds:
        rounds[round_num] = RoundResult(counts={t: 0 for t in all_teams})
    rounds[round_num].n_simulations += 1
    for w in winners:
        rounds[round_num].counts[w] += 1


def _make_sampler(
    abilities: TeamAbilities,
    config: PoissonConfig,
    rng: np.random.Generator,
) -> _Sampler:
    """Return a regulation-time scoreline sampler bound to *abilities* and *rng*."""
    def _sample(home: str, away: str) -> tuple[int, int]:
        lh, la = lambdas(abilities, home, away, neutral=True)
        mat = score_matrix(lh, la, rho=abilities.rho, max_goals=config.max_goals)
        return _draw_from_matrix(mat, rng)
    return _sample


def _simulate_once(
    ties: list[tuple[str, str]],
    all_teams: list[str],
    rounds: dict[int, RoundResult],
    resolver: Callable[[str, str, int], str],
) -> str:
    """Play one full single-elimination bracket; mutate *rounds*; return champion."""
    current, round_num, mc = ties, 0, 0
    while True:
        winners = [resolver(h, a, mc + i) for i, (h, a) in enumerate(current)]
        _record_round(rounds, round_num, all_teams, winners)
        mc += len(current)
        if len(winners) == 1:
            return winners[0]
        current, round_num = list(zip(winners[::2], winners[1::2])), round_num + 1


# ---------------------------------------------------------------------------
# Public: single-tie resolver
# ---------------------------------------------------------------------------


def resolve_tie(
    home: str,
    away: str,
    sampler: _Sampler,
    extra_time_factor: float,
    seed: int,
    abilities: TeamAbilities | None = None,
    poisson_config: PoissonConfig | None = None,
    rng: np.random.Generator | None = None,
) -> str:
    """Resolve one knockout tie: regulation → extra time → penalty coin-flip.

    *sampler* is called for regulation.  Extra-time goals use the scaled Poisson
    model when *abilities* and *rng* are supplied; otherwise *sampler* is called
    a second time (convenient for stub-based tests).  ET goals are **added** to
    the full-time score.  The penalty coin-flip is seeded by *seed*.
    """
    h, a = sampler(home, away)
    if h != a:
        return home if h > a else away
    eth, eta = _et_goals(home, away, sampler, extra_time_factor, abilities, poisson_config, rng)
    h, a = h + eth, a + eta
    if h != a:
        return home if h > a else away
    return _penalty_flip(home, away, seed)


# ---------------------------------------------------------------------------
# Standalone simulate() — public functional API
# ---------------------------------------------------------------------------


@dataclass
class KnockoutRound:
    """Per-round winner counts from the standalone simulate() function.

    ``probabilities`` holds integer advancement counts (not fractions) so that
    ``sum(final.probabilities.values()) == n_simulations`` holds exactly.
    """

    probabilities: dict[str, int]


def _play_round(
    ties: list[tuple[str, str]],
    sampler: _Sampler,
    rng: np.random.Generator,
    extra_time_factor: float,
) -> list[str]:
    """Resolve every tie in one round; derive penalty seeds from *rng*."""
    return [
        resolve_tie(h, a, sampler=sampler, extra_time_factor=extra_time_factor,
                    seed=int(rng.integers(2**32)))
        for h, a in ties
    ]


def _run_bracket(
    ties: list[tuple[str, str]],
    sampler: _Sampler,
    rng: np.random.Generator,
    extra_time_factor: float,
) -> list[list[str]]:
    """Play one full bracket; return per-round winner lists (R32 first)."""
    current, all_rounds = ties, []
    while True:
        winners = _play_round(current, sampler, rng, extra_time_factor)
        all_rounds.append(winners)
        if len(winners) == 1:
            return all_rounds
        current = list(zip(winners[::2], winners[1::2]))


def _accumulate(
    round_winners: list[list[str]],
    counts: list[dict[str, int]],
) -> None:
    """Increment per-team advancement counts for each round in-place."""
    for r, winners in enumerate(round_winners):
        if r >= len(counts):
            counts.append({})
        for w in winners:
            counts[r][w] = counts[r].get(w, 0) + 1


def simulate(
    ties: list[tuple[str, str]],
    sampler: _Sampler,
    seed: int,
    n_simulations: int = 1000,
    extra_time_factor: float = 0.33,
) -> list[KnockoutRound]:
    """Run *n_simulations* single-elimination brackets; return per-round counts.

    Each simulation uses an independent RNG seeded by ``seed + sim_idx`` so
    repeated calls with the same *seed* are fully deterministic.  Penalty seeds
    are derived from the per-simulation RNG to avoid any cross-round collision.
    """
    round_counts: list[dict[str, int]] = []
    for sim_idx in range(n_simulations):
        rng = np.random.default_rng(seed + sim_idx)
        _accumulate(_run_bracket(ties, sampler, rng, extra_time_factor), round_counts)
    return [KnockoutRound(probabilities=rc) for rc in round_counts]


# ---------------------------------------------------------------------------
# KnockoutSimulator
# ---------------------------------------------------------------------------


class KnockoutSimulator:
    """Monte Carlo single-elimination knockout simulator for WC2026.

    Seeds the R32 from live group standings via *resolve_r32*, then folds the
    bracket upward (R32 → R16 → QF → SF → Final) for each simulation, using
    the Dixon-Coles model for regulation and scaled-λ extra time.
    """

    def __init__(
        self,
        abilities: TeamAbilities,
        config: SimulationConfig,
        poisson_config: PoissonConfig | None = None,
    ) -> None:
        self._abilities = abilities
        self._config = config
        self._pcfg = poisson_config or PoissonConfig()

    def simulate(
        self,
        standings: dict[str, list[dict[str, Any]]],
        n_simulations: int | None = None,
    ) -> dict[int, RoundResult]:
        """Run *n_simulations* full knockouts; return per-round advancement counts."""
        n = n_simulations or self._config.n_simulations
        ties = resolve_r32(standings)
        all_teams = [t for pair in ties for t in pair]
        rounds: dict[int, RoundResult] = {}
        for i in range(n):
            rng = np.random.default_rng(self._config.random_seed + i)
            _simulate_once(ties, all_teams, rounds, self._make_resolver(rng, i))
        logger.info("Completed %d knockout simulations", n)
        return rounds

    def _make_resolver(
        self,
        rng: np.random.Generator,
        sim_idx: int,
    ) -> Callable[[str, str, int], str]:
        """Return a per-match resolver bound to this simulation's RNG."""
        sampler = _make_sampler(self._abilities, self._pcfg, rng)
        def _resolve(home: str, away: str, match_idx: int) -> str:
            seed = self._config.random_seed + sim_idx * 1000 + match_idx
            return resolve_tie(
                home, away, sampler=sampler,
                extra_time_factor=self._config.extra_time_factor,
                seed=seed, abilities=self._abilities,
                poisson_config=self._pcfg, rng=rng,
            )
        return _resolve
