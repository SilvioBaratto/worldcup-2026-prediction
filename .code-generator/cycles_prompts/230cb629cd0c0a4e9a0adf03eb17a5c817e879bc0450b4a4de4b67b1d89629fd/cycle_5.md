# Cycle 5 — Evaluation, Simulation & Integration

## Objective
Deliver the live forecast end-to-end: extend the tournament simulator to condition on played WC2026 results and simulate the remainder, scrape historical odds as the backtest baseline, add RPS/log-loss/Brier evaluation, and expose the new config sections and CLI commands producing full title odds.

## Project vision
A Python CLI producing a live, in-tournament probabilistic forecast of the World Cup 2026 winner — full title odds plus round-by-round advancement for all 48 teams — accuracy-first, matching or beating the bookmaker baseline, re-runnable after each matchday.

## Preceding cycles
Cycle 1 (data layer + bracket-slotting), Cycle 2 (Elo + Dixon-Coles scoreline sampler in `simulation/poisson.py`), Cycle 3 (`features/build.py`), Cycle 4 (`models/hybrid.py` primary, `models/ordered_logit.py` secondary, legacy baseline). Existing `simulation/tournament.py` (`TournamentSimulator`, `BracketSlot`, `RoundResult`) handles only fixed brackets; `models/evaluation.py` (`ModelEvaluator`) reports confusion matrix + classification report; `config.py` lacks the new sections; `cli.py` exposes `download/clean/train/fit/simulate/bracket/run`. `visualization/plots.py` (`ResultPlotter`) renders bracket + probability PNGs.

## Following cycles
None — this is the final integration cycle.

## In scope
- Extend `simulation/tournament.py`: ingest played WC2026 results as fixed; simulate remaining group matches with correct FIFA tiebreaks (goal-diff, goals-for, head-to-head, coin-flip); resolve Round of 32 from final standings via the Cycle-1 template; simulate knockout (draw → extra time `λ×0.33` → penalty coin-flip); run ~100,000 Monte Carlo tournaments → full title odds + per-round advancement for all 48 teams using the hybrid scoreline sampler.
- `data/odds.py`: scrape archived WC2014/18/22 outright + match odds, de-vig (`p_i = (1/o_i)/Σ(1/o_j)`), cache to CSV; degrade gracefully on blocked sources.
- Extend `models/evaluation.py`: RPS (primary), log-loss, Brier on time-aware splits; backtest the hybrid on WC2014/18/22 vs the scraped bookmaker baseline and vs the legacy classifier.
- `config.py`: add `[poisson]`, `[elo]`, `[rf]`, `[odds]`, `[live]` Pydantic + TOML sections.
- `cli.py`: add `fetch-live`, `build-features`, `train-hybrid`, `backtest`, `forecast` commands; keep existing `bracket`/PNG rendering and command surface.
- Title-odds visualization additions in `plots.py`.

## Out of scope
- Re-deriving Elo, Poisson abilities, or features — consume Cycles 2–3 outputs.
- Re-architecting model training — consume Cycle 4 artifacts.
- Using bookmaker odds as a model feature — evaluation baseline only.

## Acceptance criteria
- Hybrid RPS ≈ or below the bookmaker baseline on WC2014/18/22 backtests (accuracy is the acceptance criterion).
- ~100k Monte Carlo tournaments complete in seconds–low minutes; deterministic given a seed.
- `forecast` runs no-key by default and is re-runnable after each matchday; football-data.org key (in `.env`) optional for live state.
- Odds/Elo scrapers cache and degrade gracefully so a blocked source never breaks the forecast.
- Typer command surface, PNG outputs, and the existing ~248 tests stay intact; every new module adds tests.
