# Cycle 1 — Data Foundations

## Objective
Deliver the no-key data layer: a CC0 martj42 historical loader (default training source), a live WC2026 adapter over the existing `FootballClient`, a martj42↔API team-name crosswalk, and WC2026 Round-of-32 bracket-slotting rules. Every nation gets real history and the simulator can resolve `LAST_32` from group standings.

## Project vision
A Python CLI producing a live, in-tournament probabilistic forecast of the World Cup 2026 winner — full title odds plus round-by-round advancement for all 48 teams — accuracy-first, aiming to match or beat the bookmaker baseline. It replaces the legacy SVM/RF/NB classifier with a Groll-style RF hybrid (time-weighted Dixon-Coles/Poisson abilities + Elo as covariates) and conditions on results played to date.

## Preceding cycles
None. The existing repo already provides: `data/client.py` (`FootballClient`, rate-limited football-data.org v4), `data/loader.py` (`DataLoader`), `data/cleaner.py` (`DataCleaner`), `data/builders.py`, `data/bracket_builder.py` (`BracketBuilder`), and `config.py` Pydantic models. Reuse `FootballClient` for live pulls; do not rebuild it.

## Following cycles
Elo and Dixon-Coles estimation (Cycle 2), feature assembly (Cycle 3), model training (Cycle 4), simulation/evaluation/CLI wiring (Cycle 5). Do NOT compute ratings, build features, or simulate here — only land clean, normalized data and the bracket template.

## In scope
- `data/martj42_loader.py`: fetch CC0 `results.csv` (49,477 rows: `date, home_team, away_team, home_score, away_score, tournament, city, country, neutral`), coercing `"NA"` scores → nullable Int and `"TRUE"/"FALSE"` → bool; also `shootouts.csv` (678 rows) and `goalscorers.csv` (47,741 rows). Map into internal schema `DATE, HOME_TEAM, AWAY_TEAM, HOME_GOALS, AWAY_GOALS, TOURNAMENT, NEUTRAL`. martj42 already contains the WC2026 schedule.
- `data/live.py`: pull WC2026 standings, played results, and remaining group fixtures via `FootballClient` (`GET /v4/competitions/WC/matches` — 104 matches/44 played; `/standings` — 12 groups). Expose "state of tournament as of today".
- Team-name crosswalk dict (e.g. `Türkiye`/`Turkey`, `Czechia`, `South Korea`, `United States`/`USA`, `IR Iran`) joining live results to history.
- WC2026 R32 bracket-slotting: encode top-2-per-group + 8-best-thirds template and best-third assignment rules so a simulator can resolve `LAST_32`.
- Schema-verification tests against the exact data-contract shapes.

## Out of scope
- Elo engine (`data/elo.py`) — Cycle 2.
- Dixon-Coles / Poisson (`simulation/poisson.py`) — Cycle 2.
- Feature vectors (`features/build.py`) — Cycle 3.
- Odds scraping (`data/odds.py`) — Cycle 5.
- Running the live group→knockout simulation — Cycle 5.

## Acceptance criteria
- Loaders are no-key by default (martj42 + computed sources); the football-data.org key is optional and only for live state, read from `.env` (gitignored — never commit secrets).
- Schemas match the data contracts exactly (49,477 martj42 rows; 104 WC2026 matches, 44 played).
- Team names normalize so live results join historical training data.
- New modules ship with tests; the existing ~248-test suite stays green.
