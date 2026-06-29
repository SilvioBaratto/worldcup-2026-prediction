"""Single-elimination knockout simulator for WC2026.

Resolves the R32 bracket from group standings, then folds the tree
(R32 → R16 → QF → SF → Final) via Monte Carlo sampling:
  regulation → extra time (λ × extra_time_factor) → penalty coin-flip.

All knockout matches are treated as neutral-venue fixtures.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from worldcup_playoff.config import PoissonConfig, SimulationConfig
from worldcup_playoff.data.wc2026_bracket import R32_SLOTS, resolve_r32
from worldcup_playoff.simulation.poisson import (
    TeamAbilities,
    lambdas,
    score_matrix,
)

logger = logging.getLogger(__name__)


@dataclass
class RoundResult:
    """Per-team advancement counts at a knockout round.

    Each team's probability is its individual advancement probability
    (``count / n_simulations``); for rounds with multiple concurrent ties the
    probabilities across teams sum to the number of ties, not to 1.0.
    """

    counts: dict[str, int] = field(default_factory=dict)
    n_simulations: int = 0

    @property
    def probabilities(self) -> dict[str, float]:
        """Per-team advancement fractions (empty when no simulations run)."""
        if self.n_simulations == 0:
            return {}
        return {k: v / self.n_simulations for k, v in self.counts.items()}

# Re-export for consumers who import from this module.
__all__ = [
    "KnockoutRound",
    "KnockoutSimulator",
    "R32_SLOTS",
    "RoundResult",
    "representative_shootout",
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


# Penalty shootouts are close to a coin toss — a ~50-Elo edge is worth only ~52%
# (arXiv:2510.17641; FiveThirtyEight) — so any skill edge is capped near 50/50.
_PEN_MAX_TILT: float = 0.15


def _shootout_winner(
    home: str,
    away: str,
    seed: int,
    abilities: TeamAbilities | None,
    poisson_config: PoissonConfig | None,
    rng: np.random.Generator | None,
    penalty_skill: float,
) -> str:
    """Penalty-shootout winner: a small, literature-calibrated edge for the stronger
    team when *penalty_skill* > 0 and abilities are available, else a fair coin flip.

    The edge is the team's share of *decisive* regulation outcomes, tilted around
    0.5 by *penalty_skill* and clamped to ±``_PEN_MAX_TILT`` so shootouts stay
    close to random (as the evidence suggests).
    """
    if abilities is None or rng is None or penalty_skill <= 0.0:
        return _penalty_flip(home, away, seed)
    cfg = poisson_config or PoissonConfig()
    lh, la = lambdas(abilities, home, away, neutral=True)
    mat = score_matrix(lh, la, rho=abilities.rho, max_goals=cfg.max_goals)
    hw = float(np.tril(mat, -1).sum())  # P(home wins in regulation)
    aw = float(np.triu(mat, 1).sum())   # P(away wins in regulation)
    decisive = hw + aw
    edge = (hw - aw) / decisive if decisive > 0 else 0.0  # in [-1, 1]
    prob_home = 0.5 + penalty_skill * 0.5 * edge
    prob_home = min(max(prob_home, 0.5 - _PEN_MAX_TILT), 0.5 + _PEN_MAX_TILT)
    return home if float(rng.random()) < prob_home else away


def representative_shootout(seed: int, conversion: float = 0.75) -> tuple[int, int]:
    """A representative penalty-shootout score as ``(winner_kicks, loser_kicks)``.

    Best-of-five then sudden death, each kick converted with probability
    *conversion* (~75% historically). Returns the decisive tally with the winner
    first, e.g. ``(4, 3)`` or ``(5, 4)`` — used only to *display* how a tie that
    the model expects to go level was settled, not to decide who advances.
    """
    rng = np.random.default_rng(seed)
    h = a = 0
    for _ in range(5):
        h += int(rng.random() < conversion)
        a += int(rng.random() < conversion)
    while h == a:
        h += int(rng.random() < conversion)
        a += int(rng.random() < conversion)
    return (max(h, a), min(h, a))


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
    penalty_skill: float = 0.0,
) -> str:
    """Resolve one knockout tie: regulation → extra time → penalty shootout.

    *sampler* is called for regulation.  Extra-time goals use the scaled Poisson
    model when *abilities* and *rng* are supplied; otherwise *sampler* is called
    a second time (convenient for stub-based tests).  ET goals are **added** to
    the full-time score.  The shootout is a fair coin flip (seeded by *seed*)
    unless *penalty_skill* > 0 and abilities are available, in which case the
    stronger team gets a small, capped edge.
    """
    h, a = sampler(home, away)
    if h != a:
        return home if h > a else away
    eth, eta = _et_goals(home, away, sampler, extra_time_factor, abilities, poisson_config, rng)
    h, a = h + eth, a + eta
    if h != a:
        return home if h > a else away
    return _shootout_winner(home, away, seed, abilities, poisson_config, rng, penalty_skill)


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
                penalty_skill=getattr(self._config, "penalty_skill", 0.0),
            )
        return _resolve
