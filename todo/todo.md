# TODO — Upgrade to a credible WC2026 prediction model

Status of the current build, why it is biased, and a research-backed plan to fix it.
Compiled from two deep-research passes (data providers + modeling methodology).

---

## 0. Where we are now (and the problem)

Current pipeline: `download → clean → train (SVM / RandomForest / GaussianNB) → fit per-team scipy
distributions → Monte Carlo single-elimination bracket`.

It runs end-to-end, but the live result is **not a credible forecast**:

- Predicted champion came out **Switzerland (35.8%)**, with the entire top-8 European.
- Root cause = **data + method**, not code:
  - **Data:** football-data.org free tier only gave club leagues + Euro 2024 + WC2026 group
    games. Only **11 / 32** bracket nations had real fitted distributions; the other 21
    (Argentina, Brazil, Mexico, Japan…) fell back to a pooled league-average prior and washed out.
  - **Method:** a **win/loss classifier throws away goal margin** — can't produce goal
    difference (group tiebreaks), can't model draws naturally, and the GOALS-as-feature setup
    leaks the target. Goal-based models are the standard for a reason.

**Two things must change: the data source and the model.**

---

## 1. Prior work (who has done this) — reference

| Who | Method | Result / note |
|---|---|---|
| **Groll / Ley / Schauberger / Van Eetvelde** (arXiv 1806.03208) | Random Forest using **time-weighted bivariate-Poisson team-abilities as a covariate** (hybrid) + bookmaker consensus + covariates | Leading academic WC model; reproduced for 2018/2022/2026 |
| **Leitner-Zeileis-Hornik** | Bookmaker-consensus: de-vig ~24–28 books, average log-odds, inverse-simulate abilities | Predicted 2010 winner + 3/4 of 2014 semifinalists |
| **Dixon-Coles 1997** (JRSS C 46(2):265–280) | Independent Poisson + attack/defence + home advantage + **ρ low-score correction** + time-decay | Foundational — most practical models build on it |
| **Robberechts-Davis** (KU Leuven) | **Elo-difference + home-advantage ordered logit** | **Best 2002–2014** (accuracy 0.5938, RPS 0.1860) — beat bivariate Poisson, RF, bookmakers; 20k MC sims |
| **Egidi-Torelli** | Double Poisson | Beat result-classifiers on Brier (0.610 vs bookmakers 0.656) — small sample, inconclusive |
| Practical code | opisthokonta/goalmodel (R), dashee87 & pena.lt/y Dixon-Coles **Python** tutorials, Kaggle sslp23 "Predicting FIFA 2022 with ML" | Copy-ready reference implementations |

---

## 2. Football modeling logic — what to internalize

- **Poisson goal model:** team goals ~ Poisson(λ), λ = f(attack_i, defence_j, home_adv).
  Predict the **scoreline**, then derive W/D/L.
- **Dixon-Coles** adds: **ρ correction** (plain Poisson underestimates 0-0/1-0/0-1/1-1) +
  **exponential time-decay** weighting (recent matches matter more).
- **Bivariate Poisson:** models the correlation between the two teams' goals directly.
- **Elo / World Football Elo:** rating updated by result × margin × match importance;
  Elo-difference → win probability. Simple and strong.
- **Ordered logit:** treat W/D/L as an ordered outcome with Elo-diff as predictor — best in the
  Robberechts study.
- **Why goal-based ≫ the current win/loss classifier:** margin information is kept, group
  **goal-difference** tiebreaks become possible, **draws** are modeled naturally, and any matchup
  can be simulated. A plain classifier cannot do tournament standings.
- **Draws:** group stage = W/D/L + goal-diff; knockout = if drawn → extra time (λ × 0.33) then
  penalties as a coin-flip / skill term.
- **Bookmaker odds:** the **benchmark to beat** and a strong **feature** (consensus ≈ best public
  forecast).
- **Monte Carlo:** draw goals per match from the goal model → apply tournament rules → repeat
  ~100,000 times → title odds. **(We already have this layer — keep it.)**

---

## 3. How to build it — practical pipeline

- **Features:** Elo + **Elo-difference** (strongest single signal), FIFA ranking, time-weighted
  recent form, goal difference, rest days, confederation, **neutral-venue flag**, match importance.
- **Models** (best → simplest): Groll RF-hybrid > Dixon-Coles / bivariate Poisson >
  **Elo-diff ordered logit** (best effort-to-accuracy) > XGBoost on engineered features.
- **Validation:** **time-aware splits** (never shuffle chronological data). Metrics: **RPS**
  (primary), **log-loss**, **Brier** — always compared against a **bookmaker baseline**.
  Raw accuracy alone is misleading.

---

## 4. Data & providers — NO API key

| Source | Gives | Use |
|---|---|---|
| **martj42/international_results** (CC0) ⭐ | Every international match 1872→present, scores | **Primary training data** — fixes the bias (all nations get real history) |
| **eloratings.net** (World Football Elo) | Per-team Elo rating history | Key feature / second model |
| **FIFA rankings** (Kaggle cnc8 / cashncarry) | Monthly FIFA ranking points | Feature |
| **football-data.co.uk** | Club leagues + **bookmaker odds** | Odds benchmark (not international results) |
| **StatsBomb open-data** | Event-level data, some WCs | Optional advanced stats (CC-BY-NC) |
| **FBref via soccerdata / ScraperFC** | Per-match box scores incl. World Cup | Optional advanced stats (scrape, rate-limited) |
| opisthokonta/goalmodel · dashee87 · pena.lt/y | Reference Dixon-Coles implementations | Copy modeling code |

---

## 5. Recommended upgrade (the actual plan)

Replace the SVM/RF/NB win/loss classifier with a **goal-based model**, keep the Monte Carlo bracket.

- [ ] **Data loader — no key.** Add `data/martj42_loader.py` that pulls
      `results.csv` from martj42/international_results (CC0) into the existing `matches.csv` schema
      (`DATE, HOME_TEAM, AWAY_TEAM, HOME_GOALS, AWAY_GOALS, COMPETITION, neutral`).
      Replaces / supplements `FootballClient` so `clean` runs with zero auth.
- [ ] **Elo feature builder.** Add `data/elo.py` — either scrape/ingest eloratings.net history or
      compute World Football Elo from match history. Emit per-team, per-date Elo + Elo-diff.
- [ ] **Goal model.** Add `simulation/poisson.py` implementing **Dixon-Coles** (attack/defence +
      home advantage + ρ + exponential time-decay). Fit on martj42 history.
- [ ] **Elo ordered-logit baseline.** Add a simple `models/ordered_logit.py` (Elo-diff + home/neutral)
      as a strong, cheap second model / ensemble member.
- [ ] **Neutral venue.** WC2026 is mostly neutral host cities → set `home_advantage = 0` for
      neutral matches (otherwise the model invents a home edge that doesn't exist).
- [ ] **Swap the per-tie predictor.** In `simulation/game.py`, change `predict_tie` to **sample a
      scoreline from the goal model** instead of `classifier.predict`. Knockout draw → extra time
      (λ×0.33) → penalty coin-flip. Keep `TournamentSimulator` as-is.
- [ ] **Evaluation.** Add `models/evaluation.py` metrics: **RPS, log-loss, Brier** on
      **time-aware** splits, reported against a bookmaker-odds baseline (football-data.co.uk).
- [ ] **Config.** Add `[poisson]` (time-decay half-life, max goals) and `[elo]` (k-factor, init
      rating) sections; keep `[simulation] n_simulations`.
- [ ] **Re-run** full WC2026 bracket → expect a credible favorite set (Argentina / France / Spain /
      Brazil / England) instead of the European-only artifact.
- [ ] **(Optional, advanced)** Groll-style RF hybrid: feed Dixon-Coles abilities + Elo + FIFA rank +
      form as covariates into gradient boosting; blend with a bookmaker consensus.

---

## 6. Caveats (don't oversell)

- Metric wins in the literature are **small-sample** (e.g. Egidi-Torelli double-Poisson Brier edge
  is over only 16 WC2018 knockout matches; the authors call it inconclusive). On pseudo-R² that
  study ranks bookmakers above the Poisson models.
- Bookmaker counts/favorites are **year-specific** (24 books WC2026, 28 WC2022, 26 in 2018).
- The **WC2026 hybrid** result is a June 2026 r-bloggers/Zeileis blog reproduction, **not yet
  peer-reviewed**.
- The 0.33 extra-time multiplier and coin-flip shootout are **Groll et al. implementation choices**,
  not universal standards.
- "Dixon-Coles = bivariate Poisson" is loose wording: the base is **independent** Poisson with a
  low-score correction.
- Verify **licensing / coverage / freshness** of every data source before shipping
  (martj42 CC0, eloratings.net, soccerdata/ScraperFC/FBref, football-data.co.uk, StatsBomb).

---

## 7. Sources

- Groll et al., *A hybrid random forest to predict … FIFA World Cup 2018* — https://arxiv.org/pdf/1806.03208
- Zeileis, *Probabilistic forecast 2022 FIFA World Cup* — https://www.zeileis.org/news/fifa2022/
- Football meets ML — forecasting WC2026 — https://www.r-bloggers.com/2026/06/football-meets-machine-learning-forecasting-the-2026-fifa-world-cup/
- Leitner-Zeileis-Hornik bookmaker consensus — https://ideas.repec.org/p/inn/wpaper/2018-09.html
- Dixon & Coles 1997 (JRSS C) — https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/1467-9876.00065
- Egidi-Torelli comparison — https://leoegidi.github.io/paper/egidi_comparing.pdf
- Robberechts-Davis (KU Leuven) — https://lirias.kuleuven.be/server/api/core/bitstreams/7527e8cb-f047-4ef5-8395-89edd5ddc792/content
- opisthokonta goalmodel — https://github.com/opisthokonta/goalmodel · https://opisthokonta.net/?p=890
- Dixon-Coles in Python — https://dashee87.github.io/football/python/predicting-football-results-with-statistical-modelling-dixon-coles-and-time-weighting/ · https://pena.lt/y/2021/06/24/predicting-football-results-using-python-and-dixon-and-coles/
- Better metrics than RPS — https://pena.lt/y/2025/05/01/better-metrics-for-football-forecasts-moving-beyond-the-ranked-probability-score/
- Kaggle: Predicting FIFA 2022 with ML — https://www.kaggle.com/code/sslp23/predicting-fifa-2022-world-cup-with-ml
- Data: martj42 — https://github.com/martj42/international_results · World Football Elo — https://eloratings.net/ · football-data.co.uk — https://www.football-data.co.uk/data.php · StatsBomb — https://github.com/statsbomb/open-data · soccerdata — https://github.com/probberechts/soccerdata
- Other WC2026 repos — https://github.com/Hicruben/world-cup-2026-prediction-model · https://github.com/0xNadr/wc2026 · https://www.datacamp.com/tutorial/fifa-world-cup-2026-winner-prediction
