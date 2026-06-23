# World Cup 2026 Prediction — Competitive Live Forecast (Groll RF Hybrid)

## Description
A Python CLI that produces a **live, in-tournament probabilistic forecast** of the FIFA World Cup
2026 winner — full title odds for every nation plus round-by-round advancement probabilities. The
goal is **accuracy-first**: match or beat the bookmaker baseline (measured by RPS / log-loss / Brier).
It replaces the current SVM/RandomForest/NaiveBayes win/loss classifier (data-biased, European-only
artifact) with a **Groll-style Random-Forest hybrid** that uses time-weighted Dixon-Coles/Poisson team
abilities + Elo as covariates, trained on full national-team history. WC2026 is **mid-group-stage**, so
the forecast conditions on results played to date and simulates the remaining groups → knockout.

## Tech Stack
<!-- The tool uses this information to select the right agents -->
- **Language**: Python 3.11+
- **Framework**: Typer CLI + Pydantic config (library; **no web UI**)
- **Database**: none (CSV / JSON artifacts on disk)
- **Deploy**: local CLI (`worldcup-playoff`), hatchling build
- **Tests**: pytest (250 existing tests must stay green)
- **Core libs**: numpy, pandas, scikit-learn, **statsmodels** (Poisson), scipy, fitter, matplotlib,
  requests, **beautifulsoup4/lxml** (odds + Elo scraping)
- **Modeling**: Dixon-Coles / bivariate Poisson abilities → Random-Forest / gradient-boosting hybrid
  (Groll-Ley-Zeileis style), Elo ordered-logit as a secondary/baseline

## Key Decisions (locked with the user)
- **Purpose:** real competitive forecast — beat the bookies.
- **Model:** full Groll RF hybrid (Poisson abilities + Elo as RF covariates).
- **Covariates:** **football-only** — NO socio-economic / market-value / GDP. NO yfinance.
- **Bookmaker odds:** **NOT a model feature.** Used only as the **backtest baseline** to beat.
- **Output:** full title odds (every team) + round-by-round advancement.
- **Sim start:** WC2026 is in progress (Round of 16 not yet reached) → **use all data to date**:
  fold in played group results, simulate remaining group fixtures → Round of 32 → knockout.
- **Data:** martj42 (no-key history, all nations) **+** football-data.org (live WC2026).
- **Deliverable:** CLI + PNG outputs (as today).

## Features
<!-- List the features. Each entry becomes a GitHub Issue -->
1. **No-key historical loader (martj42).** `data/martj42_loader.py` — fetch CC0
   `martj42/international_results` `results.csv` (cols `date, home_team, away_team, home_score,
   away_score, tournament, city, country, neutral`; coerce `"NA"` scores → nullable, `"TRUE"/"FALSE"` →
   bool) plus `shootouts.csv` and `goalscorers.csv`. Map into the internal `matches.csv` schema
   (`DATE, HOME_TEAM, AWAY_TEAM, HOME_GOALS, AWAY_GOALS, TOURNAMENT, NEUTRAL`). Every nation gets real
   history, and the file already includes the WC2026 schedule. Default training source (zero auth).
   See **Data Contracts** for exact shapes.
2. **Live WC2026 adapter (football-data.org).** `data/live.py` — pull current WC2026 standings,
   played results, and **remaining group fixtures** via the existing `FootballClient` (key in `.env`).
   Normalize team names to match martj42. Provide "state of tournament as of today".
3. **Elo engine.** `data/elo.py` — compute World Football Elo from match history (k-factor × margin ×
   match importance) or ingest eloratings.net; emit per-team, per-date Elo + Elo-difference. Seed the
   live state from history so WC2026 teams enter with current ratings.
4. **Dixon-Coles ability estimator.** `simulation/poisson.py` — independent/bivariate Poisson with
   attack & defence strengths, home advantage, `rho` low-score correction, **exponential time-decay**.
   Fit on martj42; expose per-team attack/defence abilities as model covariates and as a standalone
   scoreline sampler.
5. **Football-only feature builder.** `features/build.py` — assemble per-match covariates: Elo,
   Elo-diff, FIFA ranking, time-weighted recent form, goal difference, Dixon-Coles attack/defence
   abilities, rest days, confederation, **neutral-venue flag**. All no-key.
6. **Groll RF hybrid model.** `models/hybrid.py` — Random-Forest / gradient-boosting regressor-or-
   classifier that consumes the Poisson abilities + Elo + covariates to predict per-match goals (or
   W/D/L distribution). This is the primary predictor. Keep `models/ordered_logit.py` (Elo-diff
   ordered logit) as a strong secondary, and the legacy SVM/RF/NB only as a backtest baseline.
7. **Live tournament simulator.** Extend `simulation/tournament.py` to: (a) ingest played WC2026
   results as fixed, (b) simulate **remaining group matches** with correct FIFA group + tiebreak
   rules to resolve the Round of 32, (c) simulate the knockout (draw → extra time `λ×0.33` → penalty
   coin-flip), (d) run ~**100,000** Monte Carlo tournaments → **full title odds + per-round
   advancement** for all 48 teams.
8. **Historical odds scraper (backtest baseline).** `data/odds.py` — scrape archived WC2014/2018/2022
   outright + match odds (de-vig to probabilities). Robust to source changes; cache to CSV. Used only
   for evaluation, never as a feature.
9. **Evaluation + backtest.** `models/evaluation.py` — **RPS (primary), log-loss, Brier** on
   **time-aware** (no-shuffle) splits; backtest the hybrid on WC2014/2018/2022 and report metrics
   **vs the scraped bookmaker baseline** and vs the legacy classifier. Success = ≈ or beat bookies.
10. **Config + CLI surface.** Add `[poisson]`, `[elo]`, `[rf]`, `[odds]`, `[live]` config sections
    (Pydantic + TOML). CLI commands: `fetch-live`, `build-features`, `train-hybrid`, `backtest`,
    `forecast` (live title odds), plus existing `bracket`/PNG rendering. Keep 250 tests green and add
    tests for every new module.

## Non-functional Requirements
<!-- Performance, security, scalability, accessibility -->
- **Accuracy is the acceptance criterion:** hybrid RPS ≈ or below the bookmaker baseline on backtests.
- **No-key default:** training/forecast must run without any API key (martj42 + computed Elo). The
  football-data.org key is optional, only for live WC2026 state; stored in `.env` (gitignored — the
  repo is public, never commit secrets).
- **Live re-runnable:** `forecast` can be re-run after each matchday to update odds as results land.
- **Reproducibility:** deterministic given a seed; no nondeterministic fitting that breaks replays.
- **Validation rigor:** time-aware splits only; never shuffle chronological data; report probabilistic
  metrics (RPS/log-loss/Brier), not raw accuracy.
- **Performance:** ~100k Monte Carlo tournaments complete in seconds–low minutes.
- **Resilience:** odds/Elo scrapers degrade gracefully (cache + fallback) so a blocked source never
  breaks the forecast.
- **Backward compatibility:** keep the Typer command surface, PNG outputs, and the existing test suite.

## Project Structure
<!-- Optional: folder structure preferences -->
```
worldcup_playoff/
├── data/
│   ├── martj42_loader.py     # NEW — no-key CC0 historical results (default training source)
│   ├── live.py               # NEW — football-data.org live WC2026 state (standings/results/fixtures)
│   ├── elo.py                # NEW — World Football Elo engine
│   ├── odds.py               # NEW — historical odds scraper (backtest baseline only)
│   ├── client.py builders.py loader.py cleaner.py bracket_builder.py   # existing
├── features/
│   └── build.py              # NEW — football-only covariate assembly
├── models/
│   ├── hybrid.py             # NEW — Groll-style RF/GBM hybrid (primary predictor)
│   ├── ordered_logit.py      # NEW — Elo-diff ordered logit (secondary)
│   ├── classifiers.py        # legacy SVM/RF/NB (backtest baseline only)
│   └── evaluation.py         # EXTEND — RPS / log-loss / Brier + backtest vs bookmakers
├── simulation/
│   ├── poisson.py            # NEW — Dixon-Coles abilities + scoreline sampler
│   ├── tournament.py         # EXTEND — live group→knockout sim from current state
│   ├── game.py distributions.py
└── visualization/plots.py    # bracket + title-odds charts
```

## Data Contracts (probed live — exact schemas, build to these)

> Verified by hitting the real endpoints/datasets on 2026-06-23. Build loaders to these exact shapes.

### martj42/international_results (CC0, no key) — PRIMARY training + WC2026 schedule
- Base: `https://raw.githubusercontent.com/martj42/international_results/master/`
- `results.csv` — **49,477 rows**, columns:
  `date, home_team, away_team, home_score, away_score, tournament, city, country, neutral`
  - `home_score`/`away_score` are **`"NA"`** for unplayed fixtures → coerce to nullable Int.
  - `neutral` is the string **`"TRUE"`/`"FALSE"`** → parse to bool.
  - **Already contains WC2026 fixtures** (`tournament == "FIFA World Cup"`, scores `NA` until played,
    `neutral == TRUE`, `country == "United States"`). So martj42 alone provides the schedule + scores.
- `shootouts.csv` — **678 rows**: `date, home_team, away_team, winner, first_shooter` (penalty outcomes;
  use to validate/parametrize the knockout shootout model).
- `goalscorers.csv` — **47,741 rows**: `date, home_team, away_team, team, scorer, minute, own_goal, penalty`.

### football-data.org v4 (key in `.env`, header `X-Auth-Token`, 10 req/min) — LIVE WC2026 only
- `GET /v4/competitions/WC/matches` → `{filters, resultSet{count,first,last,played}, competition, matches[]}`.
  - Currently **104 matches, 44 played** (mid group stage, matchday 2).
  - Each match: `id, utcDate, status, stage, group, matchday, homeTeam{id,name}, awayTeam{id,name},
    score{winner, duration, fullTime{home,away}, halfTime{home,away}}`.
  - `status` ∈ `FINISHED | IN_PLAY | PAUSED | TIMED | SCHEDULED`.
  - `stage` ∈ `GROUP_STAGE | LAST_32 | LAST_16 | QUARTER_FINALS | SEMI_FINALS | THIRD_PLACE | FINAL`
    (counts: 72/16/8/4/2/1/1). **`LAST_32` = Round of 32.**
  - `group` ∈ `GROUP_A … GROUP_L` (12 groups) or `null` for knockouts.
  - `score.winner` ∈ `HOME_TEAM | AWAY_TEAM | DRAW | null`; `fullTime.home/away` are `null` until played.
  - Filter supported: `?stage=LAST_32`.
  - **Knockout slot teams are `null`** (homeTeam.name == null) until the groups resolve — DO NOT expect
    the bracket from the API; derive it (see bracket-slotting task below).
- `GET /v4/competitions/WC/standings` → `{standings:[{stage,type,group,table[]}]}`, **12 groups**.
  - `table[]` row: `position, team{id,name}, playedGames, form, won, draw, lost, points, goalsFor,
    goalsAgainst, goalDifference` — live group tables for conditioning the sim.

### Elo — COMPUTE from martj42 (no clean eloratings.net API)
- Do **not** scrape eloratings.net. Compute World Football Elo from `results.csv` chronologically:
  `R' = R + K·G·(W − We)`, with K by match importance (Friendly < Qualifier < Continental < World Cup),
  G a goal-margin multiplier, `We = 1/(1+10^(-dr/400))` and `dr = home_elo − away_elo + home_adv`
  (`home_adv = 0` when `neutral`). Emit per-team, per-date Elo + pre-match Elo-diff.

### FIFA ranking (cnc8/fifa-world-ranking) — feature, but STALE
- `master/fifa_ranking-2020-12-10.csv` — **62,425 rows**:
  `id, rank, country_full, country_abrv, total_points, previous_points, rank_change, confederation, rank_date`.
- **Ends 2020-12-10** → for current points use a fresher pull (Kaggle `cashncarry/fifaworldranking`,
  updated) or scrape fifa.com. The `confederation` column (UEFA/CONMEBOL/CAF/AFC/CONCACAF/OFC) is a
  ready covariate either way; if rankings are unavailable, derive confederation from a static map.

### Bookmaker odds (backtest baseline only — NOT a feature)
- `football-data.co.uk` (e.g. `mmz4281/2324/E0.csv`) = **club** leagues, columns include match stats
  (`FTHG,FTAG,FTR,HS,AS,HST,AST,HC,AC…`) and 1X2 odds (`B365H/D/A, PSH/D/A, WHH/D/A, VCH…`). Useful as the
  **de-vig reference** (`p_i = (1/o_i) / Σ(1/o_j)`), NOT for WC outright odds.
- Historical **WC** outright/match odds (2014/18/22) must be **scraped** (oddsportal/archived) and cached.

### Team-name normalization (REQUIRED)
- martj42 and football-data.org names can differ (e.g. `Türkiye`/`Turkey`, `Czechia`, `South Korea`,
  `United States`/`USA`, `IR Iran`). Build a crosswalk dict so live results join to historical training.

### WC2026 bracket-slotting (REQUIRED — knockout teams are not in the API)
- 48 teams → 12 groups → **top 2 of each group + 8 best third-placed** advance to the Round of 32.
- Encode the official WC2026 R32 bracket template (which group-winner / runner-up / best-third meets
  whom) **and** the best-third-place selection/assignment rules, so the simulator can resolve `LAST_32`
  from simulated final group standings.

## Additional Notes

### Football modeling logic (what to implement)
- **Poisson goal model:** goals ~ Poisson(λ), λ = f(attack_i, defence_j, home_adv); predict scoreline → W/D/L.
- **Dixon-Coles:** add `rho` low-score correction + exponential time-decay (recent matches weigh more).
- **Elo / World Football Elo:** rating updated by result × margin × importance; Elo-diff → win prob.
- **Groll hybrid:** feed time-weighted Poisson abilities + Elo + covariates into a Random Forest /
  gradient boosting — beats either component alone (arXiv 1806.03208).
- **Why goal-based ≫ the old classifier:** keeps goal margin → enables group goal-difference tiebreaks,
  models draws, simulates any matchup. A win/loss classifier cannot produce tournament standings.
- **Neutral venue:** WC2026 host cities are neutral → `home_advantage = 0` for neutral matches.
- **Draws:** group = W/D/L + goal-diff; knockout = extra time (`λ×0.33`) then penalty coin-flip.
- **Bookmaker odds:** the benchmark to beat (de-vig consensus). Here used only for evaluation.
- **Monte Carlo:** condition on played results → simulate the rest → repeat ~100k → title odds.

### Prior work (reference methods)
- Groll/Ley/Schauberger/Van Eetvelde hybrid RF — arXiv 1806.03208; reproduced for WC2022/2026.
- Leitner-Zeileis-Hornik bookmaker consensus — predicted 2010 winner + 3/4 of 2014 semifinalists.
- Dixon-Coles 1997 (JRSS C 46(2):265–280) — foundational goal model.
- Robberechts-Davis (KU Leuven) — Elo-diff ordered logit best 2002–2014 (acc 0.5938, RPS 0.1860).
- Egidi-Torelli — double Poisson beat result-classifiers on Brier (small sample, inconclusive).

### Data stack (no-key training; key only for live)
- **martj42/international_results** (CC0) — all international matches 1872→present. **Primary training.**
- **football-data.org** (key in `.env`) — live WC2026 standings/results/remaining fixtures only.
- **eloratings.net** / computed Elo — World Football Elo features.
- **FIFA rankings** (Kaggle cnc8/cashncarry) — ranking feature.
- Historical WC **odds scraped** (2014/18/22) — backtest baseline only.

### Caveats
- Literature metric wins are small-sample (Egidi-Torelli Brier edge = 16 matches; inconclusive).
- WC2026 mostly neutral venues → standard home-advantage does not apply.
- WC2026 hybrid result is a June 2026 blog reproduction, not peer-reviewed.
- `0.33` extra-time multiplier + coin-flip shootout are implementation choices, not standards.
- Odds/Elo scrapers are fragile — cache and degrade gracefully; verify licensing before shipping.

### Sources
arXiv 1806.03208 · zeileis.org/news/fifa2022 · r-bloggers WC2026 (2026-06) · Dixon-Coles JRSS C
(1467-9876.00065) · ideas.repec.org/p/inn/wpaper/2018-09 · leoegidi.github.io egidi_comparing ·
KU Leuven Robberechts-Davis · github.com/opisthokonta/goalmodel · dashee87.github.io Dixon-Coles ·
pena.lt/y Dixon-Coles · Kaggle sslp23 FIFA2022 · github.com/martj42/international_results ·
eloratings.net · football-data.co.uk · github.com/probberechts/soccerdata · github.com/statsbomb/open-data
