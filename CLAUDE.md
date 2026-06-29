# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FIFA World Cup 2026 Knockout Prediction — a live, key-free **title-odds forecast** for all 48 nations. The engine is a statistical-probabilistic model (Elo + Dixon-Coles bivariate-Poisson, the family Opta / FiveThirtyEight use), **not** a trained ML classifier:

1. Fit **Dixon-Coles** abilities (attack/defence/ρ, time-decayed) on historical international results and blend them toward an **Elo** strength prior (`poisson.elo_prior_weight`).
2. Condition on the **real played group results**; derive the official **Round of 32** from final standings via FIFA's slotting rules + the 495-row **Annex C** third-place table.
3. Run a **Monte-Carlo** simulation of the bracket; each tie resolves **regulation (Poisson) → extra time (λ × 0.33) → penalties (50/50)**.
4. Aggregate title odds (`champion / N`) and per-round advancement, then render charts + a daily thumbnail.

The bracket runs Round of 32 → Round of 16 → Quarter-finals → Semi-finals → Final. Knockout ties are **single matches** (ET + penalties collapse into one win/loss) — no best-of-N anywhere. Only the knockout is stochastic; the group stage is fixed to the real results.

## Commands

```bash
# Install (editable, with dev tools)
pip install -e ".[dev]"

# Live title-odds forecast (no API key required; renders docs/*.png)
worldcup-playoff forecast --seed 42 --output docs   # 100k tournaments (config default)

# Calibration: tune the Elo-prior blend weight over WC2014/18/22
worldcup-playoff backtest --tune-prior
worldcup-playoff backtest                            # RF-hybrid backtest vs bookmaker baseline

# Calibration utilities
worldcup-playoff build-features
worldcup-playoff train-hybrid
worldcup-playoff fetch-live          # optional: pull live fixtures (needs API key)

# Daily 1080x1920 prediction thumbnail → thumbnails/<date>.png
python generate_thumbnail.py [--date 2026-06-29] [-o thumbnails/today.png]

# Tests / lint / types
pytest                            # all tests
pytest -k forecast                # forecast-related tests
pytest --cov=worldcup_playoff     # with coverage
ruff check . && ruff format .
mypy worldcup_playoff/

# Optional API key (live fixtures only; the forecast works fully offline)
export FOOTBALL_DATA_API_KEY="your-token-here"
```

## Architecture

The canonical path is the **live forecast** (`cli.forecast` → `simulation/live_forecast.py::run_forecast`):

1. **Abilities** — `data/martj42_loader` loads CC0 results → `simulation/poisson.fit_dixon_coles` → `poisson.blend_abilities_with_elo` with `data/elo.compute_elo`.
2. **State** — `data/live.build_state_from_results` reconstructs the WC2026 group state (official A–L group labels) from the real results (live API via `data/client` when available, else the martj42 cache).
3. **Bracket** — `simulation/group_stage` ranks the groups (FIFA tiebreaks) → `data/wc2026_bracket.resolve_r32` slots the official R32 using `data/wc2026_annexc` (Annex C table).
4. **Simulate** — `simulation/knockout.resolve_tie` folds each tie (regulation → ET → penalties); `live_forecast.LiveForecaster` runs N seeded tournaments and aggregates probabilities.
5. **Render** — `visualization/forecast_plots` (title odds, advancement, forecast bracket) + `visualization/plots.ResultPlotter` (bracket renderer); `generate_thumbnail.py` for the daily image.

A secondary **backtest/calibration** path (`models/`, `features/`, `data/odds`) tunes `elo_prior_weight` by RPS/log-loss/Brier — it does not run during a normal forecast.

### Package layout

```
worldcup_playoff/
├── __main__.py / cli.py     # Typer app + all commands: forecast, backtest, train-hybrid, build-features, fetch-live
├── config.py                # Pydantic config models (AppConfig + sub-configs)
├── data/
│   ├── martj42_loader.py    # CC0 historical international results loader
│   ├── elo.py               # World Football Elo engine
│   ├── live.py              # WC2026 tournament-state adapter (+ official A–L group labels)
│   ├── client.py            # rate-limited football-data.org v4 client (circuit breaker)
│   ├── crosswalk.py         # team-name normalization
│   ├── wc2026_bracket.py    # official group → R32 slotting (R32_SLOTS, resolve_r32)
│   ├── wc2026_annexc.py     # official 495-row Annex C third-place table
│   └── odds.py              # bookmaker odds scraper (backtest baseline)
├── simulation/
│   ├── poisson.py           # Dixon-Coles + Elo blend + lambdas/score_matrix/modal/decisive
│   ├── group_stage.py       # group-stage resolver (FIFA tiebreaks)
│   ├── knockout.py          # tie resolver (reg → ET → penalties), RoundResult, KnockoutSimulator
│   └── live_forecast.py     # Monte-Carlo forecast orchestrator (ForecastResult, LiveForecaster)
├── features/                # covariates for calibration (build, confederation, timeaware)
├── models/                  # dataset, RF/GBM hybrid, backtest evaluator
└── visualization/           # forecast_plots + ResultPlotter (NBA-style bracket)
generate_thumbnail.py        # daily 1080x1920 prediction thumbnail
```

### Key design choices

- **Statistical, not ML**: title odds come from Dixon-Coles + Elo + Monte Carlo, not a trained classifier.
- **Bracket order matters**: `R32_SLOTS` is in true bracket-adjacency order (NOT FIFA match-number order) so the knockout simulators — which pair adjacent winners — reproduce the official R16→Final tree.
- **Official Annex C**: third-place slotting uses the verbatim 495-row FIFA table (`wc2026_annexc`), not a generic bipartite matching (which finds a feasible but not FIFA's chosen assignment).
- **Neutral venue**: knockout `lambdas(..., neutral=True)` — no home advantage applied (hosts are not boosted). The `home_adv` parameter is only estimated during the fit.
- **Penalties / ET**: extra time scores at λ × `extra_time_factor` (0.33); penalties are a 50/50 coin flip. The win % includes both; the displayed "predicted score" is the most-likely **decisive** scoreline (`decisive_scoreline`).
- **Reproducible**: a fixed seed expands to N independent child seeds via `SeedSequence.spawn`, so the same seed yields identical odds.
- **`RoundResult.probabilities`** are per-team advancement (counts / n_simulations) — they sum to the number of ties in the round, not 1.0.
- **All config is Pydantic** loaded from TOML.

## Configuration

- `config/default.toml` — `[simulation]` (n_simulations=100000, extra_time_factor, seed), `[poisson]` (half-life, ρ, `elo_prior_weight`=0.8, `market_value_prior_weight`=0.0 squad-value prior), `[elo]`, `[hybrid]`/`[rf]` + `[odds]` (calibration), `[client]`/`[live]`/`[martj42]` (data sources).
- `config/playoff_2026*.toml` — the real Round of 32 bracket (reference; the forecast derives R32 from standings).
- Environment: `FOOTBALL_DATA_API_KEY` — optional football-data.org token (live fixtures only).

## Tooling

- Python 3.11+ required
- Build: hatchling
- Lint/format: ruff (line-length 100, target py311)
- Type checking: mypy (strict mode)
- Tests: pytest (testpaths = `tests/`)
