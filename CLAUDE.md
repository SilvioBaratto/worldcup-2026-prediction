# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FIFA World Cup 2026 Knockout Prediction — Monte Carlo simulation of World Cup 2026 knockout outcomes using ML classifiers (SVM, Random Forest, Gaussian Naive Bayes) trained on historical match statistics (2006 onward). Statistical distributions are fitted to each national team's recent performance, then thousands of knockout brackets are simulated by sampling synthetic match features and predicting outcomes.

The bracket runs Round of 32 → Round of 16 → Quarter-finals → Semi-finals → Final (48 teams qualify; 32 reach the knockout stage). Unlike the NBA project this is adapted from, **knockout ties are single matches** — extra time and penalties collapse into one win/loss outcome — so there is no best-of-N series anywhere in the code.

## Commands

```bash
# Install (editable, with dev tools)
pip install -e ".[dev]"

# Provide an API key (optional — unauthenticated works at the lower 10 req/min rate)
export FOOTBALL_DATA_API_KEY="your-token-here"

# Run full pipeline (clean → train → fit → simulate → visualize)
worldcup-playoff run --bracket config/playoff_2026.toml

# Individual steps
worldcup-playoff clean
worldcup-playoff train --classifier all          # or: svm, random-forest, naive-bayes
worldcup-playoff fit
worldcup-playoff simulate --bracket config/playoff_2026.toml --n-simulations 10000
worldcup-playoff bracket --bracket config/playoff_2026.toml   # simulate + render PNG

# Download datasets from football-data.org (rate-limited)
worldcup-playoff download --seasons 2006-2026 --output-dir dataset/csv
worldcup-playoff download --only matches,teams --seasons 2018-2026
worldcup-playoff download --skip-details          # skip match_details.csv

# Build individual datasets
worldcup-playoff build-teams
worldcup-playoff build-matches --start-year 2006 --end-year 2026
worldcup-playoff build-match-details --matches-csv dataset/csv/matches.csv

# Generate a knockout bracket from the competition's qualified teams
worldcup-playoff generate-bracket --season 2026 --output config/playoff_2026.toml

# Without installation
python -m worldcup_playoff run --bracket config/playoff_2026.toml

# Tests
pytest                            # all tests
pytest tests/test_game.py         # single file
pytest -k test_predict            # single test by name
pytest --cov=worldcup_playoff     # with coverage

# Linting & formatting
ruff check .
ruff format .
```

## Architecture

**Pipeline pattern**: `Pipeline` (in `pipeline.py`) orchestrates stages, each independently runnable via the Typer CLI (`cli.py`, exposed as the `worldcup-playoff` entry point `worldcup_playoff.cli:app`):

1. **Clean** — `DataLoader` reads raw CSVs → `DataCleaner` merges `matches.csv` with `match_details.csv` on `MATCH_ID`, filters by date, drops draws, fixes zero percentages, adds the binary `HOME_WIN` target → produces `dataset/train_data.csv`
2. **Train** — `ClassifierFactory` creates sklearn classifiers → `ClassifierTrainer` fits (temporal split, `shuffle=False`) → `ModelEvaluator` reports metrics → models saved to `output/models/*.joblib`
3. **Fit** — `DistributionFitter` fits scipy distributions to each team's per-feature data (using the `fitter` library) → saved to `output/distributions.json`
4. **Simulate** — `FeatureSampler` draws synthetic stats from the fitted distributions → `GamePredictor` predicts each single-match tie → `TournamentSimulator` runs N full brackets via Monte Carlo → `ResultPlotter` renders output

### Package layout

```
worldcup_playoff/
├── __main__.py         # Enables `python -m worldcup_playoff`
├── cli.py              # Typer commands — all user-facing entry points (worldcup_playoff.cli:app)
├── config.py           # Pydantic models (AppConfig, BracketConfig, Matchup) loaded from TOML
├── pipeline.py         # Orchestrator — wires stages together, only class that touches all subpackages
├── types.py            # Classifier Protocol (sklearn interface contract)
├── data/
│   ├── client.py       # FootballClient — rate-limited football-data.org v4 client with
│   │                   #   circuit-breaker retry + session reset; X-Auth-Token from env
│   ├── builders.py     # CSV builders (TeamsBuilder, MatchesBuilder, RankingBuilder,
│   │                   #   PlayersBuilder, MatchDetailsBuilder) — each fetches from football-data.org
│   ├── bracket_builder.py  # Generates a Round-of-32 bracket TOML from qualified teams
│   ├── loader.py       # Reads local CSVs into DataFrames with column + dtype validation
│   └── cleaner.py      # Merges matches + details, engineers features, filters dates, adds HOME_WIN
├── models/
│   ├── classifiers.py  # ClassifierFactory (creates SVM/RF/NB), ClassifierTrainer (fit/save/load)
│   └── evaluation.py   # ModelEvaluator — classification report + confusion matrix + ROC curves
├── simulation/
│   ├── distributions.py # DistributionFitter (fitter→scipy), FeatureSampler, FittedDistribution dataclass
│   ├── game.py          # GamePredictor — samples features, runs classifier, returns single-tie winner
│   └── tournament.py    # TournamentSimulator — Monte Carlo bracket, BracketSlot tree, RoundResult
└── visualization/
    └── plots.py         # ResultPlotter — bracket PNG and round-probability charts
```

### Key design choices

- **Single-match ties throughout**: `GamePredictor.predict_tie(home, away)` runs one prediction per tie; `TournamentSimulator._play_tie` resolves a tie with a single call (no `_play_series` loop). `SimulationConfig` has no series-length fields.
- **Dependency injection throughout**: `GamePredictor` receives classifier + sampler + distributions; `Pipeline` receives config objects. No global state.
- **`Classifier` Protocol** in `types.py` — any object with `fit(X, y)` and `predict(X)` works.
- **All config is Pydantic models** loaded from TOML (`config/default.toml` for pipeline params, `config/playoff_*.toml` for bracket matchups).
- **`FootballClient`** uses a circuit-breaker pattern (3 consecutive failures → 60 s cooldown), exponential backoff with jitter, periodic session resets, and a 6 s default delay tuned for the free 10 req/min tier. The API key is read from `FOOTBALL_DATA_API_KEY` and sent as `X-Auth-Token`; the client still works unauthenticated.
- **Deterministic feature heuristics**: `MatchDetailsBuilder` fills shots/possession/pass-accuracy with fixed formulae (`SHOTS = goals*5+7`, `SHOTS_ON_TARGET = max(goals+2, shots//3)`, `POSSESSION = 50.0`, `PASS_PCT = 75.0`) when the free tier omits them — never random, so output is reproducible. Paid-tier statistics override the heuristics.
- **`RoundResult.probabilities`** are per-team advancement probabilities (counts / n_simulations), not normalized across teams — they sum to the number of ties in the round, not 1.0.
- **Bracket tree** (`BracketSlot`) is built bottom-up from matchups via `build_bracket_tree`; adjacent winners pair into the next round. Matchup-list length must be a power of two (32-team knockout → 16 first-round matchups).
- **Team identification by name**: `matches.csv` stores country names directly in `HOME_TEAM` / `AWAY_TEAM`, so the cleaner needs no numeric-ID-to-name mapping (unlike the NBA original). `DistributionFitter` aggregates per-team observations by combining `*_home` columns from home matches with `*_away` columns from away matches.

## Configuration

- `config/default.toml` — pipeline parameters (data paths, 10 feature columns, classifier hyperparams, distribution candidates, simulation settings, client rate-limit config)
- `config/playoff_2026.toml` — bracket definition (16 Round-of-32 matchups with `home` / `away` / `group`); adjacent matchups feed the next round, list length must be a power of two
- Environment: `FOOTBALL_DATA_API_KEY` — optional football-data.org token (sent as `X-Auth-Token`)

## Tooling

- Python 3.11+ required
- Build: hatchling
- Lint/format: ruff (line-length 100, target py311)
- Type checking: mypy (strict mode)
- Tests: pytest (testpaths = `tests/`)
