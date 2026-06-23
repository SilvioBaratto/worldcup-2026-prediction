## Goal
Implement `worldcup_playoff/simulation/poisson.py` — a **Dixon-Coles / bivariate-Poisson estimator** fitted on martj42 history. It exposes (a) per-team attack & defence abilities (plus home advantage, `rho`, intercept) as model covariates, and (b) a standalone deterministic scoreline sampler (`goals ~ Poisson(λ)` with the Dixon-Coles low-score correction). The abilities feed the Cycle 3 feature builder; the sampler feeds the Cycle 5 Monte Carlo tournament.

## Context
The normalized data layer already exists (Cycle 1, merged on `main`). `worldcup_playoff/data/martj42_loader.py` → `Martj42Loader.load_results()` returns a DataFrame with **`DATE`** (object/ISO string — NOT datetime), **`HOME_TEAM`**, **`AWAY_TEAM`** (already crosswalk-normalized), **`HOME_GOALS`**/**`AWAY_GOALS`** (nullable `Int64`, `<NA>` for unplayed fixtures), **`TOURNAMENT`**, **`NEUTRAL`** (bool). Consume this frame directly.

Dixon-Coles model (1997):
- `λ_home = exp(intercept + attack_home − defence_away + home_adv)`, `λ_away = exp(intercept + attack_away − defence_home)`
- Low-score correction `τ(h, a, λ_home, λ_away, ρ)`: `(0,0) → 1 − λ_home·λ_away·ρ`; `(0,1) → 1 + λ_home·ρ`; `(1,0) → 1 + λ_away·ρ`; `(1,1) → 1 − ρ`; else `1.0`. With `ρ = 0` the model reduces to independent Poisson.
- **Exponential time-decay**: weight `w_i = 0.5 ^ (age_days_i / half_life_days)`, `age_days_i = (reference_date − match_date).days`, reference defaults to the max `DATE` in the fitted data — recent matches weigh more.
- Fit by **weighted MLE** with `scipy.optimize.minimize` (`statsmodels` is NOT a project dependency; `scipy>=1.11` is). The fit must be **deterministic** (fixed start values → L-BFGS-B is deterministic). Enforce a mean-zero attack normalization for identifiability.
- `home_adv` is dropped (set to 0) when a match is neutral — WC2026 host cities are neutral (mirror the Elo neutral handling).

Conventions to follow (observed in `simulation/distributions.py`, `config.py`): `from __future__ import annotations`; frozen `@dataclass` for value objects (mirror `FittedDistribution`); Pydantic `BaseModel` + `ConfigDict(extra="ignore")` + `@field_validator` for config; dependency injection (config into constructor, no globals); module-level factory functions (mirror `load_martj42_results` / `fit_*`); `logger = logging.getLogger(__name__)`; `|` unions. The sampler's determinism must mirror `FeatureSampler` in `distributions.py`: `np.random.default_rng(random_state)`, reproducible with the same seed, divergent across seeds. mypy strict, ruff line-length 100. The module must be **I/O-free / network-free**; tests build DataFrames in-memory. The existing ~411-test suite must stay green.

This task also adds the minimal `[poisson]` Pydantic config section — only what the estimator/sampler need to run (full TOML wiring is Cycle 5). It touches `config.py` additively (one new class + one `AppConfig` field); if the sibling Elo task already edited `config.py`, add your `poisson:` field without disturbing theirs.

**Scope guard:** expose the abilities and sampler only. Do NOT assemble feature vectors (Cycle 3), train the hybrid (Cycle 4), or run the tournament Monte Carlo (Cycle 5).

## Agent and skills
- **Agents:** `python-pro`
- **Active skills:** `solid-principles`, `python-expert`

## Acceptance criteria
- [ ] `dixon_coles_tau(h, a, λ_home, λ_away, ρ)` returns the exact correction for the four ≤1 scorelines and `1.0` otherwise; `ρ = 0` ⇒ all `τ == 1.0`; `τ > 0` for valid small `ρ` with default λ (rho-validity check).
- [ ] `DixonColesEstimator.fit(df)` returns `TeamAbilities(attack, defence, home_adv, rho, intercept)`, fitting only on **played** matches (rows with `<NA>` goals are dropped by `_prepare`; fit never raises on their presence).
- [ ] Time-decay (half-life in **DAYS**): a match `half_life_days` old has half the weight of a fresh one; `2·half_life` old → quarter weight; the most-recent match (= reference date) → weight 1.0 (`pytest.approx`).
- [ ] Fit is **deterministic**: fitting twice on identical data yields identical abilities (tight `approx`).
- [ ] Fitted `attack` values satisfy the mean-zero normalization (`approx(0)`) — identifiability.
- [ ] Recovery sanity (seeded synthetic data): a clearly stronger team gets a higher fitted `attack`.
- [ ] `ScorelineSampler.sample(home, away, neutral=False, size, random_state)` returns an int array of shape `(size, 2)` with goals in `[0, max_goals]`; reproducible with the same seed, divergent across seeds; default seed comes from `PoissonConfig`.
- [ ] `lambdas(abilities, home, away, neutral=True)` drops `home_adv` ⇒ strictly lower `λ_home` than the non-neutral case.
- [ ] `score_matrix(...)` (joint pmf grid) sums to 1.0 (`approx`) and equals the product of marginals when `ρ = 0`.
- [ ] `PoissonConfig` defaults load; invalid values (`half_life_days <= 0`, `max_goals < 1`, `optimizer_maxiter < 1`) raise `ValidationError`; `AppConfig().poisson` is a `PoissonConfig`.
- [ ] All tests pass
- [ ] SOLID, clean code (methods < 10 lines, classes < 50 lines, files < 500–600 lines), TDD

## Dependencies
- None — consumes the Cycle 1 data layer (`data/martj42_loader.py`, `config.py`) already merged on `main`. Independent of the Elo task (shares only an additive `config.py` edit).

## Implementation notes
- **Create:** `worldcup_playoff/simulation/poisson.py`
  - Frozen dataclass `TeamAbilities(attack: dict[str, float], defence: dict[str, float], home_adv: float, rho: float, intercept: float)`.
  - Module-level pure function `dixon_coles_tau(...)` (the τ correction, no state).
  - `DixonColesEstimator(config: PoissonConfig)` with small methods: `_prepare(df)` (drop `<NA>` goals via `.isna()`, `pd.to_datetime`, compute decay weights, build team index, extract integer goal arrays), `_neg_log_likelihood(params, ...)` (vectorized weighted DC log-likelihood using `scipy.stats.poisson.logpmf`), `fit(df) -> TeamAbilities` (pack params → `scipy.optimize.minimize(method="L-BFGS-B", options={"maxiter": optimizer_maxiter})` → unpack → mean-zero normalize), `lambdas(abilities, home, away, neutral=False)`.
  - `ScorelineSampler(abilities, config)` with `sample(...)` building the `(max_goals+1)²` joint-pmf grid with the τ correction, normalizing, and drawing via `rng.choice` on the flattened categorical (`np.random.default_rng(random_state or config.random_seed)`); plus `score_matrix(...)` returning the normalized grid.
  - Module-level factories: `fit_dixon_coles(df, config=None)` and `make_sampler(abilities, config=None)`.
- **Modify:** `worldcup_playoff/config.py` — add `class PoissonConfig(BaseModel)` (`ConfigDict(extra="ignore")`) with: `half_life_days: float = 365.0` (document the DAYS unit), `max_goals: int = 10`, `rho_init: float = -0.1`, `home_adv_init: float = 0.25`, `min_matches: int = 1`, `random_seed: int = 42`, `optimizer_maxiter: int = 200`, plus `@field_validator`s (`half_life_days > 0`, `max_goals >= 1`, `optimizer_maxiter >= 1`). Register `poisson: PoissonConfig = PoissonConfig()` on `AppConfig` (additive — preserve any sibling `elo:` field).
- **Create:** `tests/test_poisson.py` — BDD `test_when_..._then_...` naming; in-memory synthetic DataFrames (no network/I/O); parametrize the τ scorelines; seed all RNG for reproducibility; cover every acceptance criterion above.
