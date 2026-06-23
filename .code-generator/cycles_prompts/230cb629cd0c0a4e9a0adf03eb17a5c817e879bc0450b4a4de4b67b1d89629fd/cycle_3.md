# Cycle 3 — Feature Assembly

## Objective
Build `features/build.py`: assemble per-match, football-only covariate vectors combining Elo, Dixon-Coles abilities, FIFA ranking/confederation, recent form, goal difference, rest days, and the neutral-venue flag — for both historical training matches and current WC2026 state.

## Project vision
A Python CLI producing a live, in-tournament probabilistic forecast of the World Cup 2026 winner — full title odds plus round-by-round advancement for all 48 teams — accuracy-first, aiming to match or beat the bookmaker baseline, via a Groll-style RF hybrid conditioned on results to date.

## Preceding cycles
Cycle 1 delivered the data layer (`data/martj42_loader.py`, `data/live.py`, crosswalk, bracket-slotting). Cycle 2 delivered `data/elo.py` (per-team, per-date Elo + Elo-diff) and `simulation/poisson.py` (Dixon-Coles attack/defence abilities). This cycle reads those rating/ability outputs — do not recompute Elo or Poisson fits.

## Following cycles
Cycle 4 trains the hybrid and baselines on these features; Cycle 5 simulates and evaluates. Do NOT train any model, define `[rf]` hyperparameters, or run backtests here.

## In scope
- `features/build.py`: a feature builder that emits, per match, the covariates: Elo, Elo-difference, FIFA ranking (with confederation as fallback when ranking is stale/unavailable), time-weighted recent form, goal difference, Dixon-Coles attack/defence abilities, rest days, confederation (UEFA/CONMEBOL/CAF/AFC/CONCACAF/OFC), and neutral-venue flag.
- Assemble feature frames from martj42 (historical) and from the live WC2026 adapter (current state), joined via the Cycle-1 crosswalk.
- FIFA-ranking handling: cnc8 data ends 2020-12-10 → use a fresher pull or derive confederation from a static map; ranking absence must degrade to the confederation covariate, not fail.
- Tests for feature shapes, neutral-flag correctness, fallback behavior, and time-aware (no-leakage) construction.

## Out of scope
- Any socio-economic / market-value / GDP / yfinance covariates — explicitly excluded by locked decisions.
- Bookmaker odds as a feature — odds are backtest-baseline only (Cycle 5).
- Computing Elo or Dixon-Coles abilities — Cycle 2 (consume, don't recompute).
- Model fitting and the RF hybrid — Cycle 4.

## Acceptance criteria
- All covariates are football-only; no socio-economic/market-value features and no odds.
- Builder runs no-key (martj42 + computed Elo/Poisson + static confederation map); live state is optional and key-gated.
- Features are constructed time-aware with no forward leakage; deterministic given a seed.
- Feature frames cover both historical training matches and current WC2026 matches.
- New module ships with tests; the existing ~248-test suite stays green.
