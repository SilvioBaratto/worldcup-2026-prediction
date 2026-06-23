# Cycle 4 — Model Training & Comparison

## Objective
Train the primary Groll-style RF/GBM hybrid on the Cycle-3 covariates, add an Elo-diff ordered-logit secondary, and retain the legacy SVM/RF/NB classifiers solely as a backtest baseline — all fit with time-aware (no-shuffle) splits.

## Project vision
A Python CLI producing a live, in-tournament probabilistic forecast of the World Cup 2026 winner — full title odds plus round-by-round advancement for all 48 teams — accuracy-first, aiming to match or beat the bookmaker baseline. The hybrid (Poisson abilities + Elo + covariates) is the primary predictor; the ordered logit a strong secondary.

## Preceding cycles
Cycle 1 (data layer), Cycle 2 (`data/elo.py`, `simulation/poisson.py`), Cycle 3 (`features/build.py` covariate frames). Legacy models already exist in `models/classifiers.py` (`ClassifierFactory`, `ClassifierTrainer` — SVM/RF/NB) and `models/evaluation.py` (`ModelEvaluator`); reuse, do not rebuild, those classifiers as the baseline.

## Following cycles
Cycle 5 runs the live tournament simulation, scrapes odds, computes RPS/log-loss/Brier backtests, adds `[poisson]/[elo]/[rf]/[odds]/[live]` config, and wires the new CLI commands. Do NOT scrape odds, run Monte Carlo, or add CLI commands here.

## In scope
- `models/hybrid.py`: Random-Forest / gradient-boosting hybrid (Groll-Ley-Zeileis style, arXiv 1806.03208) consuming Poisson abilities + Elo + Cycle-3 covariates to predict per-match goals (or a W/D/L distribution). This is the primary predictor — goal-based so it preserves goal margin for group tiebreaks and models draws.
- `models/ordered_logit.py`: Elo-diff ordered logit (statsmodels) as a strong secondary/fallback.
- Keep legacy SVM/RF/NB in `models/classifiers.py` available as the backtest baseline only.
- Time-aware training: chronological splits, `shuffle=False`; deterministic given a seed.
- New tests for the hybrid and ordered-logit modules.

## Out of scope
- RPS/log-loss/Brier scoring and the WC2014/18/22 backtest harness — Cycle 5 (`models/evaluation.py` extension).
- Historical odds scraping (`data/odds.py`) — Cycle 5.
- Tournament/Monte-Carlo simulation and title-odds output — Cycle 5.
- New CLI commands (`train-hybrid`, `backtest`, `forecast`) and `[rf]` TOML section finalization — Cycle 5 (this cycle may stub config needed to fit, but command surface is Cycle 5).

## Acceptance criteria
- Hybrid is goal-based (or full W/D/L distribution), enabling group goal-difference tiebreaks and draws.
- All models train with time-aware splits; never shuffle chronological data.
- Deterministic given a seed; no nondeterministic fitting that breaks replays.
- The full pipeline runs no-key (martj42 + computed Elo/Poisson features).
- New hybrid + ordered-logit modules ship with tests; the existing ~248-test suite stays green.
