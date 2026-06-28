"""Live-tournament Monte-Carlo forecast orchestrator for WC2026.

Runs ~100k seeded simulations (group-stage → knockout) conditioned on played
WC2026 results and aggregates per-team title odds + per-round advancement
probabilities for all 48 teams.

Key design choices
------------------
- **Callable injection**: ``LiveForecaster`` accepts ``group_simulator`` and
  ``knockout_simulator`` protocol callables so it can be tested with stubs and
  used with real ``GroupStageSimulator`` / ``KnockoutSimulator`` via factory
  adapters.  This decouples orchestration from the concrete implementations in
  ``group_stage.py`` and ``knockout.py``.
- **SeedSequence sub-seeding**: the master seed is expanded into N
  statistically-independent child seeds via ``numpy.random.SeedSequence.spawn``
  so that (a) the same master seed yields bit-identical odds on any re-run, and
  (b) each tournament iteration draws from an independent, decorrelated stream —
  avoiding the intra-tournament correlation that arises from re-seeding with the
  same integer.
- **Fit-once pattern**: ``TeamAbilities`` are computed outside the Monte-Carlo
  loop and accepted by ``LiveForecaster.run()``; the loop only draws samples.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any, Callable

import numpy as np

from worldcup_playoff.data.live import TournamentState
from worldcup_playoff.simulation.poisson import TeamAbilities

logger = logging.getLogger(__name__)

# Stable round order: used by visualisations to render columns R32 → Final.
WC_ROUND_ORDER: tuple[str, ...] = ("R32", "R16", "QF", "SF", "Final")

# Re-export so callers can import everything from one place.
__all__ = [
    "ForecastResult",
    "LiveForecaster",
    "TournamentState",
    "TeamAbilities",
    "WC_ROUND_ORDER",
]

# ---------------------------------------------------------------------------
# Callable protocols (structural — no ABC overhead)
# ---------------------------------------------------------------------------

#: Signature: (state, abilities, rng) → list of 32 qualified team names.
_GroupSimFn = Callable[
    [TournamentState, TeamAbilities, np.random.Generator],
    list[str],
]

#: Signature: (qualified_teams, abilities, rng) → {"champion": str, "rounds": dict}.
_KnockoutSimFn = Callable[
    [list[str], TeamAbilities, np.random.Generator],
    dict[str, Any],
]

# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ForecastResult:
    """Immutable result of a live Monte-Carlo forecast.

    Attributes
    ----------
    champion_probabilities:
        Per-team title-odds: ``count / n_simulations``.  **All 48 group-stage
        teams are present**, including those with probability 0.0.
    round_probabilities:
        ``{round_name: {team: probability}}`` advancement fractions.
        Reuses the ``RoundResult.probabilities`` contract: each team's value
        is its individual advancement probability (``count / n_simulations``),
        so values across a round sum to the number of ties in that round, not
        to 1.0.  Only rounds that appear in at least one simulation are
        included.
    """

    champion_probabilities: dict[str, float]
    round_probabilities: dict[str, dict[str, float]]
    #: A single representative Round-of-32 draw (16 (home, away) pairs) from one
    #: deterministic group-stage resolution, used to lay out the bracket plot.
    #: Empty when no qualifiers could be resolved.
    representative_r32: tuple[tuple[str, str], ...] = ()
    #: Expected goals (xG) per representative slot pairing, keyed ``"HOME|AWAY"``
    #: -> ``(xg_home, xg_away)`` — the Dixon-Coles goal rates λ, rounded to one
    #: decimal. Preferred over the modal scoreline, whose mode collapses to 1-0
    #: for almost every low-scoring match. Empty when no draw is available.
    representative_xg: dict[str, tuple[float, float]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Private helpers (module-level, ≤10 lines each)
# ---------------------------------------------------------------------------


def _teams_from_matches(matches: list[Any]) -> set[str]:
    """Collect non-null home/away team names from a flat list of match objects."""
    teams: set[str] = set()
    for m in matches:
        if m.home_team:
            teams.add(m.home_team)
        if m.away_team:
            teams.add(m.away_team)
    return teams


def _extract_all_teams(state: TournamentState) -> list[str]:
    """Return sorted list of all group-stage teams from standings or match history."""
    teams = {
        row.team_name
        for gs in state.standings
        for row in gs.table
        if row.team_name
    }
    if not teams:
        teams = _teams_from_matches(list(state.played) + list(state.remaining_group_fixtures))
    return sorted(teams)


def _accumulate_rounds(
    acc: dict[str, dict[str, int]],
    rounds: dict[str, Any],
) -> None:
    """Merge one simulation's per-round team-count dict into *acc* in place."""
    for round_name, team_counts in rounds.items():
        bucket = acc.setdefault(round_name, {})
        for team, count in team_counts.items():
            bucket[team] = bucket.get(team, 0) + count


def _to_probabilities(counts: dict[str, int], n: int) -> dict[str, float]:
    """Divide raw counts by *n*; return zeroes when n == 0."""
    if n == 0:
        return {t: 0.0 for t in counts}
    return {t: c / n for t, c in counts.items()}


# ---------------------------------------------------------------------------
# Representative bracket topology (shared with the bracket visualization)
# ---------------------------------------------------------------------------


def favourite(teams: list[str], probs: dict[str, float]) -> str:
    """Most-likely team in *teams* by advancement probability (name breaks ties)."""
    return max(teams, key=lambda t: (probs.get(t, 0.0), t)) if teams else "TBD"


def forecast_slot_teams(
    r32: list[tuple[str, str]],
    round_probabilities: dict[str, dict[str, float]],
) -> dict[int, list[tuple[str, str]]]:
    """Round index -> (home, away) pairs for the representative bracket.

    Round 0 is the representative R32 draw. Each later-round box shows the
    favourite of each of its two feeding sub-brackets, ranked by that round's
    advancement probability, so the likeliest team fills each downstream slot.
    """
    slots: dict[int, list[tuple[str, str]]] = {0: [(h, a) for h, a in r32]}
    under: list[list[str]] = [[h, a] for h, a in r32]
    rnd = 1
    while len(under) > 1:
        name = WC_ROUND_ORDER[min(rnd, len(WC_ROUND_ORDER) - 1)]
        probs = round_probabilities.get(name, {})
        pairs: list[tuple[str, str]] = []
        merged: list[list[str]] = []
        for i in range(0, len(under) - 1, 2):
            left, right = under[i], under[i + 1]
            pairs.append((favourite(left, probs), favourite(right, probs)))
            merged.append(left + right)
        slots[rnd] = pairs
        under = merged
        rnd += 1
    return slots


def _representative_xg(
    abilities: TeamAbilities,
    r32: tuple[tuple[str, str], ...],
    round_probabilities: dict[str, dict[str, float]],
) -> dict[str, tuple[float, float]]:
    """Expected goals (Dixon-Coles λ, 1 dp) for every representative slot pairing.

    Expected goals are used instead of the modal scoreline because the mode of a
    low-scoring match collapses to 1-0 almost everywhere, hiding the differences
    between fixtures; the λ rates vary smoothly with team strength.
    """
    from worldcup_playoff.simulation.poisson import lambdas
    if not r32:
        return {}
    slots = forecast_slot_teams(list(r32), round_probabilities)
    xg: dict[str, tuple[float, float]] = {}
    for pairs in slots.values():
        for home, away in pairs:
            if home and away and home != "TBD" and away != "TBD":
                lh, la = lambdas(abilities, home, away, neutral=True)
                xg[f"{home}|{away}"] = (round(lh, 1), round(la, 1))
    return xg


# ---------------------------------------------------------------------------
# LiveForecaster
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# High-level CLI entry + concrete adapter helpers
# ---------------------------------------------------------------------------


def _fetch_state_and_abilities(cfg: Any) -> tuple[TournamentState, TeamAbilities]:
    """Load tournament state and fit Dixon-Coles abilities (may raise).

    Tries the live football-data.org API first; if it is unreachable (e.g. no API
    key → HTTP 403, or offline), falls back to reconstructing the WC2026 group
    state from the martj42 cache so ``forecast`` still runs key-free.
    """
    from worldcup_playoff.data.live import build_state_from_results, fetch_tournament_state
    from worldcup_playoff.data.martj42_loader import load_martj42_results
    from worldcup_playoff.simulation.poisson import blend_abilities_with_elo, fit_dixon_coles
    df = load_martj42_results(cfg.martj42)
    abilities = fit_dixon_coles(df, cfg.poisson)
    weight = getattr(cfg.poisson, "elo_prior_weight", 0.0)
    if weight > 0.0:
        from worldcup_playoff.data.elo import compute_elo
        elo = compute_elo(df, getattr(cfg, "elo", None))
        abilities = blend_abilities_with_elo(abilities, elo.final_ratings, weight)
        logger.info("Blended Dixon-Coles abilities with Elo prior (weight=%.2f).", weight)
    try:
        state = fetch_tournament_state()
    except Exception:
        logger.warning("Live API unreachable; building WC2026 state from martj42 cache.")
        state = build_state_from_results(df)
    return state, abilities


def _play_round(
    pairs: list[tuple[str, str]],
    abilities: TeamAbilities,
    sampler: Any,
    rng: np.random.Generator,
    extra_time_factor: float,
    pcfg: Any,
) -> list[str]:
    from worldcup_playoff.simulation.knockout import resolve_tie
    return [
        resolve_tie(h, a, sampler=sampler, extra_time_factor=extra_time_factor,
                    seed=int(rng.integers(2**32)), abilities=abilities, poisson_config=pcfg, rng=rng)
        for h, a in pairs
    ]


def _knockout_sim_fn(abilities: TeamAbilities, cfg: Any) -> _KnockoutSimFn:
    """Return a one-shot knockout callable wired to the real Dixon-Coles model."""
    from typing import cast
    from worldcup_playoff.config import SimulationConfig, PoissonConfig
    from worldcup_playoff.simulation.knockout import _make_sampler
    sim_cfg: SimulationConfig = cast(SimulationConfig, getattr(cfg, "simulation", SimulationConfig()))
    pcfg: PoissonConfig = cast(PoissonConfig, getattr(cfg, "poisson", PoissonConfig()))

    def _sim(qualified: list[str], ab: TeamAbilities, rng: np.random.Generator) -> dict[str, Any]:
        sampler = _make_sampler(ab, pcfg, rng)
        pairs = [(qualified[i], qualified[i + 1]) for i in range(0, len(qualified), 2)]
        rounds: dict[str, dict[str, int]] = {}
        current = _play_round(pairs, ab, sampler, rng, sim_cfg.extra_time_factor, pcfg)
        rounds["R32"] = {w: 1 for w in current}
        for rnd in ("R16", "QF", "SF", "Final"):
            if len(current) <= 1:
                break
            current = _play_round(
                [(current[i], current[i + 1]) for i in range(0, len(current), 2)],
                ab, sampler, rng, sim_cfg.extra_time_factor, pcfg,
            )
            rounds[rnd] = {w: 1 for w in current}
        return {"champion": current[0] if current else "", "rounds": rounds}

    return _sim


def _group_sim_fn(abilities: TeamAbilities) -> _GroupSimFn:
    """Return a group-stage callable wired to GroupStageSimulator + resolve_r32."""
    from worldcup_playoff.simulation.poisson import make_sampler
    from worldcup_playoff.simulation.group_stage import GroupStageSimulator
    from worldcup_playoff.data.wc2026_bracket import resolve_r32
    import random

    def _sim(state: TournamentState, ab: TeamAbilities, rng: np.random.Generator) -> list[str]:
        py_rng = random.Random(int(rng.integers(2**32)))
        gs = GroupStageSimulator(make_sampler(ab), py_rng, np_rng=rng)
        return [t for pair in resolve_r32(gs.simulate(state)) for t in pair]

    return _sim


def run_forecast(
    cfg: Any = None,
    *,
    seed: int = 42,
    n_simulations: int = 10_000,
) -> ForecastResult | None:
    """High-level CLI entry: fetch live WC2026 state, fit abilities, run N simulations.

    Falls back to ``None`` when martj42 data or the live API is unavailable so
    the ``forecast`` CLI command can degrade gracefully without crashing.
    Re-runnable: each call derives fresh child seeds from *seed* via SeedSequence.
    """
    from worldcup_playoff.config import AppConfig
    resolved = cfg if cfg is not None else AppConfig()
    try:
        state, abilities = _fetch_state_and_abilities(resolved)
    except Exception:
        return None
    result = LiveForecaster(
        _group_sim_fn(abilities), _knockout_sim_fn(abilities, resolved)
    ).run(state, abilities, n_simulations, seed)
    xg = _representative_xg(
        abilities, result.representative_r32, result.round_probabilities,
    )
    return replace(result, representative_xg=xg)


class LiveForecaster:
    """Orchestrates N seeded Monte-Carlo WC2026 tournament simulations.

    One child ``numpy.random.Generator`` is derived per iteration from the
    master seed via ``numpy.random.SeedSequence.spawn`` so that:
    - two calls with the same master seed produce bit-identical title odds;
    - draws within each tournament advance a **single** RNG, eliminating the
      intra-tournament correlation caused by re-seeding per match.
    """

    def __init__(
        self,
        group_simulator: _GroupSimFn,
        knockout_simulator: _KnockoutSimFn,
    ) -> None:
        self._group_sim = group_simulator
        self._knockout_sim = knockout_simulator

    def run(
        self,
        state: TournamentState,
        abilities: TeamAbilities,
        n_simulations: int,
        seed: int,
    ) -> ForecastResult:
        """Run *n_simulations* tournaments and return aggregated probabilities."""
        all_teams = _extract_all_teams(state)
        champion_counts: dict[str, int] = {t: 0 for t in all_teams}
        round_counts: dict[str, dict[str, int]] = {}
        child_seeds = np.random.SeedSequence(seed).spawn(n_simulations)
        for child_seed in child_seeds:
            rng = np.random.default_rng(child_seed)
            self._run_one(state, abilities, rng, champion_counts, round_counts)
        representative = self._representative_r32(state, abilities, seed)
        return self._to_result(champion_counts, round_counts, n_simulations, representative)

    def _representative_r32(
        self, state: TournamentState, abilities: TeamAbilities, seed: int
    ) -> tuple[tuple[str, str], ...]:
        """Resolve one deterministic Round-of-32 draw for the bracket layout."""
        try:
            rng = np.random.default_rng(np.random.SeedSequence(seed))
            qualified = self._group_sim(state, abilities, rng)
        except Exception:
            return ()
        return tuple(
            (qualified[i], qualified[i + 1]) for i in range(0, len(qualified) - 1, 2)
        )

    def _run_one(
        self,
        state: TournamentState,
        abilities: TeamAbilities,
        rng: np.random.Generator,
        champion_counts: dict[str, int],
        round_counts: dict[str, dict[str, int]],
    ) -> None:
        qualified = self._group_sim(state, abilities, rng)
        result = self._knockout_sim(qualified, abilities, rng)
        champ = result["champion"]
        champion_counts[champ] = champion_counts.get(champ, 0) + 1
        _accumulate_rounds(round_counts, result.get("rounds", {}))

    def _to_result(
        self,
        champion_counts: dict[str, int],
        round_counts: dict[str, dict[str, int]],
        n: int,
        representative_r32: tuple[tuple[str, str], ...] = (),
    ) -> ForecastResult:
        champ_probs = _to_probabilities(champion_counts, n)
        round_probs = {
            rnd: _to_probabilities(counts, n) for rnd, counts in round_counts.items()
        }
        return ForecastResult(
            champion_probabilities=champ_probs,
            round_probabilities=round_probs,
            representative_r32=representative_r32,
        )
