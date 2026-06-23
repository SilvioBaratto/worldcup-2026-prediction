# Cycle 2 — Statistical Abilities

## Objective
Compute the two ability systems that feed every downstream model: a chronological World Football Elo engine and a Dixon-Coles bivariate-Poisson estimator with time-decay. Both fit on martj42 history and emit per-team, per-date ratings/abilities, seeding WC2026 teams with current values.

## Project vision
A Python CLI producing a live, in-tournament probabilistic forecast of the World Cup 2026 winner — full title odds plus round-by-round advancement for all 48 teams — accuracy-first, aiming to match or beat the bookmaker baseline. It replaces the legacy classifier with a Groll-style RF hybrid (Dixon-Coles/Poisson abilities + Elo as covariates), conditioned on results to date.

## Preceding cycles
Cycle 1 delivered the normalized data layer: `data/martj42_loader.py` (history in `DATE, HOME_TEAM, AWAY_TEAM, HOME_GOALS, AWAY_GOALS, TOURNAMENT, NEUTRAL` schema, plus shootouts/goalscorers), `data/live.py` (WC2026 state), the team-name crosswalk, and the WC2026 R32 bracket-slotting template. Consume those normalized frames here — do not re-parse raw CSVs.

## Following cycles
Feature assembly (Cycle 3) consumes these ratings; model training (Cycle 4); simulation/evaluation (Cycle 5). Do NOT assemble per-match feature vectors, train the hybrid, or run Monte Carlo here.

## In scope
- `data/elo.py`: compute World Football Elo chronologically from martj42 — `R' = R + K·G·(W − We)`, `We = 1/(1+10^(-dr/400))`, `dr = home_elo − away_elo + home_adv`, with K by match importance (Friendly < Qualifier < Continental < World Cup), G a goal-margin multiplier, and `home_adv = 0` when `neutral`. Emit per-team, per-date Elo and pre-match Elo-difference. Seed WC2026 teams with their latest historical rating.
- `simulation/poisson.py`: Dixon-Coles / bivariate Poisson with per-team attack & defence strengths, home advantage, `rho` low-score correction, and exponential time-decay (recent matches weigh more). Fit on martj42; expose per-team attack/defence abilities as covariates AND a standalone scoreline sampler (goals ~ Poisson(λ)).
- Tests for Elo update math, neutral-venue handling, time-decay weighting, and `rho` correction.

## Out of scope
- FIFA-ranking ingestion, recent-form, rest-days, confederation features — Cycle 3 (`features/build.py`).
- The RF/GBM hybrid and ordered logit — Cycle 4 (`models/hybrid.py`, `models/ordered_logit.py`).
- Tournament simulation using the scoreline sampler — Cycle 5.
- `[elo]`/`[poisson]` TOML config wiring beyond what these modules need to run — finalized in Cycle 5.

## Acceptance criteria
- Both estimators run no-key from martj42 history alone.
- Deterministic given a seed; no nondeterministic fitting that breaks replays.
- `home_advantage = 0` for neutral matches (WC2026 host cities are neutral).
- Per-team, per-date Elo + Elo-diff and Dixon-Coles attack/defence abilities are emitted for downstream consumption; WC2026 live state is seeded from history.
- New modules ship with tests; the existing ~248-test suite stays green.
