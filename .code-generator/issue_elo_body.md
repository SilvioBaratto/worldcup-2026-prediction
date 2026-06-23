## Goal
Implement `worldcup_playoff/data/elo.py` — a **World Football Elo engine** that computes per-team Elo ratings chronologically from martj42 history. It emits (a) per-team, per-date ratings, (b) a pre-match Elo-difference for every match, and (c) a seed of each WC2026 team's latest historical rating. These ratings are a primary covariate for the Cycle 3 feature builder and the Cycle 4 hybrid model.

## Context
The normalized data layer already exists (Cycle 1, merged on `main`). `worldcup_playoff/data/martj42_loader.py` → `Martj42Loader.load_results()` returns a DataFrame with columns **`DATE`** (object/ISO string — NOT datetime), **`HOME_TEAM`**, **`AWAY_TEAM`** (already crosswalk-normalized), **`HOME_GOALS`**/**`AWAY_GOALS`** (nullable `Int64`, `<NA>` for unplayed fixtures incl. WC2026), **`TOURNAMENT`** (object), **`NEUTRAL`** (numpy bool). Consume this frame directly — do not re-parse raw CSVs.

The World Football Elo update is:
- `R' = R + K · G · (W − We)`
- `We = 1 / (1 + 10^(−dr/400))`, `dr = home_elo − away_elo + home_adv`
- `home_adv = 0` when `NEUTRAL` is True (WC2026 host cities are neutral)
- `W` = 1.0 home win / 0.5 draw / 0.0 home loss
- `K` by match importance, classified from `TOURNAMENT` (Friendly < Qualifier < Continental < World Cup)
- `G` is a goal-margin multiplier (World-Football-Elo style: 1.0 for margin ≤ 1, 1.5 for 2, `(11 + margin)/8` for ≥ 3)

Conventions to follow (observed in `martj42_loader.py`, `simulation/distributions.py`, `config.py`): `from __future__ import annotations`; frozen `@dataclass` for value objects (mirror `FittedDistribution`); Pydantic `BaseModel` + `ConfigDict(extra="ignore")` + `@field_validator` for config; dependency injection (config passed into the constructor, no globals); module-level factory functions wrapping the class (mirror `load_martj42_results`); `logger = logging.getLogger(__name__)`; `|` unions, not `Optional`. mypy strict, ruff line-length 100. The module must be **I/O-free and network-free** so tests build DataFrames in-memory (mirror `tests/test_martj42_loader.py` / `tests/conftest.py`). The existing ~411-test suite must stay green. `statsmodels` is NOT a dependency (not needed here); numpy/pandas/scipy are available.

This task also adds the minimal `[elo]` Pydantic config section — only what the engine needs to run (full TOML wiring is Cycle 5). It touches `config.py` additively (one new class + one `AppConfig` field); if the sibling Dixon-Coles task already edited `config.py`, add your `elo:` field without disturbing theirs.

**Scope guard:** do NOT assemble per-match feature vectors (Cycle 3), train any model (Cycle 4), or run Monte Carlo (Cycle 5).

## Agent and skills
- **Agents:** `python-pro`
- **Active skills:** `solid-principles`, `python-expert`

## Acceptance criteria
- [ ] `EloEngine.run(results_df)` computes Elo chronologically (stable sort by date, deterministic same-day tiebreak) and returns per-team/per-date ratings + per-match pre-match Elo-diff + final per-team ratings.
- [ ] Expectation math is exact: equal ratings on a neutral venue → `We == 0.5`; `dr = 400` → `We ≈ 0.909` (`pytest.approx`).
- [ ] Update formula verified against a hand-computed case (e.g. both 1500, home wins 1–0, friendly K, neutral → home 1510 / away 1490) and is zero-sum (`Δhome == −Δaway`).
- [ ] `home_advantage = 0` when `NEUTRAL` is True; non-neutral matches add `home_advantage` to `dr` (covered by a test comparing the two).
- [ ] Goal-margin multiplier `G`: margins {0,1}→1.0, 2→1.5, 3→1.75, 5→2.0.
- [ ] `K` classified by importance from `TOURNAMENT` (case-insensitive substring), with precedence so "FIFA World Cup qualification" → qualifier tier, "FIFA World Cup" → world-cup tier, a continental keyword → continental tier, else friendly.
- [ ] Unplayed fixtures (`<NA>` goals) cause **no** rating change but a pre-match Elo-diff is still emitted; engine never raises on `<NA>`.
- [ ] A team with no prior matches enters at `initial_rating` (the "no history" edge case).
- [ ] Chronological determinism: shuffling input row order yields identical final ratings.
- [ ] `seed_wc2026(result, teams)` returns each team's latest historical rating, defaulting to `initial_rating` for teams absent from history.
- [ ] `EloConfig` defaults load; invalid values (non-positive K / initial_rating, negative home_advantage) raise `ValidationError`; `AppConfig().elo` is an `EloConfig`.
- [ ] All tests pass
- [ ] SOLID, clean code (methods < 10 lines, classes < 50 lines, files < 500–600 lines), TDD

## Dependencies
- None — consumes the Cycle 1 data layer (`data/martj42_loader.py`, `data/crosswalk.py`, `config.py`) already merged on `main`.

## Implementation notes
- **Create:** `worldcup_playoff/data/elo.py`
  - Frozen dataclasses: `EloRating(team, date, rating)`; `MatchEloDiff(date, home_team, away_team, home_elo, away_elo, elo_diff, neutral)` where `elo_diff = home_elo − away_elo` (pre-match, **venue-neutral** — keep the `home_advantage` term internal to the expectation calc so the emitted covariate is venue-agnostic; document this); `EloResult(history, match_diffs, final_ratings)` with convenience `history_frame()` / `latest_ratings_frame()` helpers (keep core data in plain structures for testability).
  - `EloEngine(config: EloConfig)` with small (<10-line) methods: `_expected(home_elo, away_elo, neutral)`, `_goal_multiplier(margin)`, `_k_for_tournament(tournament)`, `_update_pair(...)`, and `run(df)` orchestrator. `run` must `pd.to_datetime(df["DATE"], errors="coerce")`, sort stably, maintain a `dict[str, float]` of current ratings defaulting to `initial_rating`, skip `<NA>`-goal rows for updates (use `.isna()` on the `Int64` columns) while still emitting their pre-match diff.
  - `seed_wc2026(result, teams: Iterable[str]) -> dict[str, float]` — pass team names explicitly (e.g. from `wc2026_schedule(df)` or a `TournamentState`) so the module stays decoupled from `live.py`/the HTTP client.
  - Module-level factory: `compute_elo(df, config: EloConfig | None = None) -> EloResult`.
- **Modify:** `worldcup_playoff/config.py` — add `class EloConfig(BaseModel)` (`ConfigDict(extra="ignore")`) with: `initial_rating: float = 1500.0`, `home_advantage: float = 100.0`, tiered K factors (`k_friendly=20`, `k_qualifier=30`, `k_continental=40`, `k_world_cup=60`), and data-driven keyword lists for tournament classification (`world_cup_keywords`, `continental_keywords`, `qualifier_keywords`). Add `@field_validator`s (positive K / initial_rating, non-negative home_advantage). Register `elo: EloConfig = EloConfig()` on `AppConfig` (additive — preserve any sibling `poisson:` field).
- **Create:** `tests/test_elo.py` — BDD `test_when_..._then_...` naming; build small DataFrames in-memory (no network/I/O); cover every acceptance criterion above; parametrize the `G` and `K` cases; assert determinism with shuffled input.
