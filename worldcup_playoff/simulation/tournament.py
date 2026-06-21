"""Monte Carlo single-elimination tournament simulation.

Mirrors ``nba_playoff.simulation.tournament`` but replaces the best-of-N series
with a **single-match** knockout tie throughout.  Key differences from the NBA
original:

- ``TournamentSimulator._play_series`` is replaced by ``_play_tie``, which
  delegates to ``GamePredictor.predict_tie`` for one prediction instead of
  accumulating wins across multiple games.
- ``SimulationConfig`` carries no ``games_per_series`` / ``wins_to_advance``
  fields; the constructor only stores ``_n_simulations`` for logging purposes.
- ``build_bracket_tree`` accepts ``list[Matchup]`` where ``Matchup.group``
  replaces ``Matchup.conference`` (field name in the football config).
- Round indices remain 0-based: 0 = Round of 32, …, 4 = Final.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

from worldcup_playoff.config import Matchup, SimulationConfig
from worldcup_playoff.simulation.game import GamePredictor

logger = logging.getLogger(__name__)


@dataclass
class RoundResult:
    """Tracks per-team advancement counts at a given round.

    Each team's probability is its individual advancement probability:
    ``count / n_simulations``.  For rounds with multiple concurrent ties this
    means the probabilities across all teams sum to the number of ties in the
    round, not to 1.0.

    Attributes:
        counts: Raw advancement counts keyed by team name.
        n_simulations: Total number of bracket simulations completed so far for
            this round (incremented once per ``_simulate_once`` call).
    """

    counts: dict[str, int] = field(default_factory=dict)
    n_simulations: int = 0

    @property
    def probabilities(self) -> dict[str, float]:
        """Convert raw counts to per-team advancement probabilities.

        Uses ``n_simulations`` as the denominator so each value correctly
        reflects the fraction of simulations in which that team advanced,
        regardless of how many ties were played concurrently in the round.

        Returns:
            Empty dict when ``n_simulations`` is 0 to avoid division by zero.
        """
        if self.n_simulations == 0:
            return {}
        return {k: v / self.n_simulations for k, v in self.counts.items()}


@dataclass
class BracketSlot:
    """A node in the bracket tree.

    Leaves hold the two teams from a first-round tie.
    Internal nodes aggregate all teams reachable through their sub-bracket.

    Attributes:
        teams: All team names that could appear in this slot.
        children: The two child ``BracketSlot`` nodes whose winners meet here,
            or ``None`` for leaf nodes.
        group: The group/conference label inherited from the originating
            ``Matchup.group`` value (propagated upward only when both children
            share the same label).
    """

    teams: list[str]
    children: tuple[BracketSlot, BracketSlot] | None = None
    group: str = ""


def build_bracket_tree(matchups: list[Matchup]) -> BracketSlot:
    """Build a bracket tree from the sequential matchup list.

    Mirrors the pairing logic in ``TournamentSimulator._simulate_once``:
    adjacent matchup winners pair into the next round until only the root
    (the Final) remains.

    Args:
        matchups: A non-empty list whose length is a power of 2.  World Cup
            knockout brackets satisfy this constraint (2, 4, 8, 16, 32 teams).

    Returns:
        The root ``BracketSlot`` whose ``teams`` contains every participant.

    Raises:
        ValueError: If ``matchups`` is empty or its length is not a power of 2.
    """
    n = len(matchups)
    if n == 0 or (n & (n - 1)) != 0:
        raise ValueError(
            f"matchups must be a non-empty power-of-2 length, got {n}"
        )

    # Build leaf nodes — one per first-round tie.
    slots: list[BracketSlot] = [
        BracketSlot(
            teams=[m.home, m.away],
            group=m.group,
        )
        for m in matchups
    ]

    # Fold adjacent pairs upward until only the root remains.
    while len(slots) > 1:
        next_level: list[BracketSlot] = []
        for i in range(0, len(slots), 2):
            left, right = slots[i], slots[i + 1]
            shared_group = left.group if left.group == right.group else ""
            next_level.append(
                BracketSlot(
                    teams=left.teams + right.teams,
                    children=(left, right),
                    group=shared_group,
                )
            )
        slots = next_level

    return slots[0]


def extract_bracket_slots(root: BracketSlot) -> dict[int, list[BracketSlot]]:
    """Return ``{round_number: [slots]}`` by BFS from root.

    Round 0 = leaves (first round), increasing toward the root (Final).

    Args:
        root: The root ``BracketSlot`` returned by ``build_bracket_tree``.

    Returns:
        Dict mapping round index to the list of ``BracketSlot`` nodes at that
        depth level.
    """
    depth = 0
    node = root
    while node.children:
        depth += 1
        node = node.children[0]

    levels: dict[int, list[BracketSlot]] = {}
    queue: deque[tuple[BracketSlot, int]] = deque([(root, 0)])
    while queue:
        slot, d = queue.popleft()
        round_num = depth - d
        levels.setdefault(round_num, []).append(slot)
        if slot.children:
            queue.append((slot.children[0], d + 1))
            queue.append((slot.children[1], d + 1))

    return levels


class TournamentSimulator:
    """Runs Monte Carlo simulations of a single-elimination knockout bracket.

    Uses composition: holds a ``GamePredictor`` rather than inheriting from it.

    Unlike the NBA ``TournamentSimulator``, there are no series-length fields.
    Each tie is decided by a single ``GamePredictor.predict_tie`` call.
    """

    def __init__(
        self,
        game_predictor: GamePredictor,
        config: SimulationConfig,
    ) -> None:
        self._predictor = game_predictor
        # Store n_simulations only for logging; the caller controls it via
        # the simulate() argument so the same instance can be reused.
        self._default_n_simulations = config.n_simulations

    def simulate(
        self,
        bracket: list[Matchup],
        n_simulations: int,
        progress_callback: Callable[[int], None] | None = None,
    ) -> dict[int, RoundResult]:
        """Run *n_simulations* full single-elimination brackets.

        Args:
            bracket: The first-round matchup list.  Length must be a power of 2
                (validated inside ``_simulate_once`` via the pairing logic).
            n_simulations: Number of independent bracket simulations to run.
            progress_callback: Optional callable invoked after each simulation
                with the 1-based iteration index, e.g. for a progress bar.

        Returns:
            ``{round_number: RoundResult}`` with accumulated advancement counts.
            Round numbers are 0-based (0 = first knockout round, N = Final).

        Raises:
            ValueError: If *bracket* is empty or its length is not a power of 2.
        """
        n = len(bracket)
        if n == 0 or (n & (n - 1)) != 0:
            raise ValueError(
                f"bracket length must be a non-empty power of 2, got {n}"
            )

        all_teams = [m.home for m in bracket] + [m.away for m in bracket]
        rounds: dict[int, RoundResult] = {}

        for i in range(n_simulations):
            self._simulate_once(bracket, all_teams, rounds)
            if progress_callback is not None:
                progress_callback(i + 1)

        logger.info("Completed %d simulations", n_simulations)
        return rounds

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _simulate_once(
        self,
        bracket: list[Matchup],
        all_teams: list[str],
        rounds: dict[int, RoundResult],
    ) -> str:
        """Play one full bracket and mutate *rounds* to accumulate counts.

        Args:
            bracket: First-round matchup list (length is a power of 2).
            all_teams: Flat list of every participating team; used to initialize
                the ``RoundResult.counts`` dict on the first simulation.
            rounds: Mutable accumulator updated in-place.

        Returns:
            The name of the tournament champion.
        """
        current_matchups = bracket
        round_num = 0

        while True:
            winners = [self._play_tie(m) for m in current_matchups]

            if round_num not in rounds:
                rounds[round_num] = RoundResult(
                    counts={team: 0 for team in all_teams}
                )
            rounds[round_num].n_simulations += 1

            for winner in winners:
                rounds[round_num].counts[winner] += 1

            # Single champion — tournament complete.
            if len(winners) == 1:
                return winners[0]

            # Pair adjacent winners for the next round.
            current_matchups = [
                Matchup(home=winners[i], away=winners[i + 1])
                for i in range(0, len(winners), 2)
            ]
            round_num += 1

    def _play_tie(self, matchup: Matchup) -> str:
        """Decide a single knockout tie via the ``GamePredictor``.

        This replaces the NBA ``_play_series`` loop entirely — one prediction,
        one winner.

        Args:
            matchup: The tie to resolve.

        Returns:
            The advancing team's name.
        """
        return self._predictor.predict_tie(matchup.home, matchup.away)
