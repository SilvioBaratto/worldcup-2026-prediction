# FIFA World Cup 2026 Knockout Prediction

A live, key-free **title-odds forecast** for all 48 nations, built on an **Elo + Dixon-Coles bivariate-Poisson** model and a Monte-Carlo simulation of the official WC2026 knockout bracket.

![FIFA World Cup 2026 — forecast bracket with champion](docs/bracket.png)

*Representative knockout bracket with per-round advancement probabilities and the projected champion (seed 42, 100,000 simulations, `elo_prior_weight = 0.8`). The ranked title-odds leaderboard (`docs/title_odds.png`) and a per-round heatmap (`docs/advancement.png`) are produced alongside it.*

## Overview

The forecast is a statistical-probabilistic engine (the same family Opta / FiveThirtyEight use), **not** a trained ML classifier. It runs in five steps:

1. **Team strength** — fit a **Dixon-Coles bivariate-Poisson** model (attack/defence/ρ, time-decayed) to all historical international results, and blend the abilities toward an **Elo** strength prior so reputable sides are not understated.
2. **Real table** — the group stage is conditioned on the **actual played results**; the official **Round of 32** is derived from the final standings via FIFA's slotting rules and the 495-row **Annex C** third-place table.
3. **Monte Carlo** — replay the real bracket *N* times; each tie is resolved **regulation (Poisson) → extra time (λ × 0.33) → penalties (50/50)**.
4. **Aggregate** — title odds = `champion count / N`; per-round advancement = `count / N`.
5. **Visualize** — title-odds leaderboard, advancement heatmap, and the NBA-style forecast bracket; plus a daily 1080×1920 prediction thumbnail.

Knockout ties are decided by a **single match** (extra time + penalties collapse into one win/loss), so there are no best-of-N series anywhere.

## Live Title-Odds Forecast

```bash
# Live title-odds forecast (no API key required; renders PNG charts)
worldcup-playoff forecast --seed 42 --output docs   # 100k tournaments (config default)

# Tune the Elo-prior blend weight by RPS / log-loss / Brier over past World Cups
worldcup-playoff backtest --tune-prior
```

The engine is conditioned on the group results played to date and run through **100,000 Monte Carlo tournaments** (the config default: at 100k the sampling 95% CI on title odds is ≈ ±0.3%, below which the model/calibration uncertainty dominates). The fitted attack/defence abilities — which on raw goals over-reward sides that ran up big group-stage scorelines — are blended toward an Elo strength prior (`poisson.elo_prior_weight`, default `0.8`). The weight was tuned by match-level **RPS** over WC2014/18/22 (pure goals = 0.2195 → `0.8` = 0.2082, ~5% better).

The `forecast` command writes three charts to `--output`: `bracket.png` (the knockout tree with per-round advancement % and a champion banner), `title_odds.png` (ranked leaderboard), and `advancement.png` (per-round heatmap).

## Installation

```bash
pip install -e ".[dev]" --config-settings editable_mode=compat
```

Python 3.11+ required. No API key is needed — the forecast falls back to the bundled martj42 (CC0) results cache when the live football-data.org API is unreachable.

### API key (optional)

```bash
export FOOTBALL_DATA_API_KEY="your-token-here"
```

Only used to pull live fixtures/standings; the forecast works fully offline without it.

## Quick Start

```bash
# Title-odds forecast → docs/*.png
worldcup-playoff forecast --seed 42 --output docs

# Daily prediction thumbnail (1080×1920) → thumbnails/<date>.png
python generate_thumbnail.py
```

## CLI Reference

All commands are exposed via the `worldcup-playoff` entry point.

### `forecast`

Run the live WC2026 title-odds forecast (no API key required). Renders `bracket.png`, `title_odds.png`, `advancement.png`.

| Flag | Default | Description |
|------|---------|-------------|
| `--config` / `-c` | `config/default.toml` | Config (sets `poisson.elo_prior_weight`, sim count) |
| `--seed` | `42` | Random seed (re-runnable; same seed → identical odds) |
| `--n-simulations` / `-n` | from config (100000) | Monte Carlo iterations |
| `--output` / `-o` | `cfg.visualization.output_dir` | Directory for PNG charts |
| `--no-plots` | off | Skip writing PNG charts |

### `backtest`

Time-aware WC backtest (RPS / log-loss / Brier) of the RF hybrid vs the bookmaker baseline. With `--tune-prior`, sweep `poisson.elo_prior_weight` over WC2014/18/22 and report the best weight.

| Flag | Default | Description |
|------|---------|-------------|
| `--tune-prior` | off | Sweep `elo_prior_weight` over past WCs and report RPS/log-loss/Brier |

### `build-features` · `train-hybrid` · `fetch-live`

Calibration/utility commands: assemble the football covariate matrix, fit the RF/GBM hybrid goal model, and (optionally) pull live fixtures from football-data.org.

### `generate_thumbnail.py`

```bash
python generate_thumbnail.py                 # next upcoming match day (auto-advances)
python generate_thumbnail.py --date 2026-06-29
python generate_thumbnail.py -o thumbnails/today.png
```

Renders a 1080×1920 brutalist story image of the day's Round-of-32 prediction(s): predicted (decisive) scoreline, winner, and advance probability.

## Architecture

```
martj42 results (CC0)  ─┐
                        ├─► Dixon-Coles fit ──► blend with Elo prior ──► TeamAbilities
World Football Elo  ────┘                                                    │
                                                                             ▼
real group results ──► standings ──► resolve_r32 (Annex C) ──► Monte-Carlo knockout
                                                                             │
                                          regulation → extra time → penalties │
                                                                             ▼
                                        title odds + per-round advancement + charts
```

Only the knockout is stochastic; the group stage is fixed to the real results, so every simulation starts from the same official Round of 32.

## Project Structure

```
worldcup_playoff/
├── cli.py / cli_cycle5.py   # Typer commands: forecast, backtest, train-hybrid, build-features, fetch-live
├── config.py                # Pydantic config models
├── data/
│   ├── martj42_loader.py    # CC0 historical international results
│   ├── elo.py               # World Football Elo engine
│   ├── live.py              # WC2026 tournament-state adapter (+ official A–L group labels)
│   ├── client.py            # rate-limited football-data.org v4 client
│   ├── crosswalk.py         # team-name normalization
│   ├── wc2026_bracket.py    # official group → R32 slotting
│   ├── wc2026_annexc.py     # official 495-row Annex C third-place table
│   └── odds.py              # bookmaker odds scraper (backtest baseline)
├── simulation/
│   ├── poisson.py           # Dixon-Coles + Elo blend + scoreline helpers
│   ├── group_stage.py       # group-stage resolver (FIFA tiebreaks)
│   ├── knockout.py          # tie resolver (reg → ET → penalties), RoundResult
│   └── live_forecast.py     # Monte-Carlo forecast orchestrator
├── features/                # covariates for calibration (build, confederation, timeaware)
├── models/                  # dataset, RF/GBM hybrid, backtest evaluator
└── visualization/           # forecast_plots + NBA-style bracket renderer
generate_thumbnail.py        # daily 1080×1920 prediction thumbnail
config/
├── default.toml             # forecast + calibration parameters
└── playoff_2026*.toml        # real Round of 32 bracket (reference)
```

## Configuration

All parameters live in `config/default.toml`:

- `[simulation]` — `n_simulations` (100000), `extra_time_factor` (0.33), `random_seed`.
- `[poisson]` — Dixon-Coles half-life, `max_goals`, ρ, and `elo_prior_weight` (0.8).
- `[elo]` — initial rating, home advantage, per-competition K-factors.
- `[hybrid]` / `[rf]` — RF/GBM hybrid goal model (backtest/calibration).
- `[odds]` — bookmaker odds scraper (backtest baseline).
- `[client]` / `[live]` / `[martj42]` — data sources.

## Modelling Assumptions & Limits

- Goals ~ **Dixon-Coles bivariate Poisson** (ρ corrects only low scores).
- Team strength is **time-decayed** (recent matches weigh more) and **static within the tournament**.
- Knockout matches are **neutral-venue** (no home advantage applied — hosts are not boosted).
- **Extra time** scores at 1/3 of the regulation rate; **penalties** are a fair 50/50 coin flip.
- The displayed "predicted score" is the most-likely **decisive** scoreline (the win % already includes ET/penalties).
- Calibration is tuned on only 3 past World Cups — wide confidence intervals.

## Results

`worldcup-playoff forecast` produces the bracket at the top of this README, the ranked title-odds leaderboard (`docs/title_odds.png`), and a per-round advancement heatmap (`docs/advancement.png`). With `elo_prior_weight = 0.8` the top tier — Argentina, Spain, France, England, Brazil — matches expert/bookmaker consensus.

## Testing

```bash
pytest                            # all tests
pytest -k forecast                # forecast-related tests
pytest --cov=worldcup_playoff     # with coverage
```

Pure unit tests — no network calls; the forecast runs against the bundled martj42 cache.

## Tooling

| Tool | Config |
|------|--------|
| **ruff** | line-length 100, target Python 3.11 |
| **mypy** | strict mode, Python 3.11 |
| **pytest** | testpaths = `tests/` |
| **hatchling** | build backend |

## License

MIT
