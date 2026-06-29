"""Model evaluation and comparison.

Public API
----------
Pure metric functions (source-blind, no sklearn dependency):
  rank_probability_score  — ordered 3-class RPS (primary)
  multiclass_log_loss     — mean cross-entropy
  brier_score             — mean multi-class Brier score

Orchestration:
  backtest_hybrid         — time-aware WC backtest vs bookmaker + legacy baselines

Legacy (kept for pipeline compatibility):
  ModelEvaluator          — confusion matrix + classification report
  plot_roc_curves         — ROC curve visualisation
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
)

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

_EPS: float = 1e-15
_DEFAULT_WC_YEARS: list[int] = [2014, 2018, 2022]
_META_COLS: frozenset[str] = frozenset(
    {"date", "home_team", "away_team", "tournament", "outcome"}
)

# ── Pure metric functions ────────────────────────────────────────────────────


def rank_probability_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean RPS for ordered 3-class (Win/Draw/Loss) predictions.

    RPS_i = (1/(J-1)) * Σ_{k=0}^{J-2} (CDF_pred_{i,k} − CDF_obs_{i,k})²
    averaged over N matches (J=3 → divides by 2).
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred, dtype=float)
    cdf_pred = np.cumsum(yp, axis=1)[:, :-1]          # (N, J-1)
    cutpoints = np.arange(yp.shape[1] - 1)            # [0, 1]
    cdf_obs = (yt[:, np.newaxis] <= cutpoints).astype(float)
    return float(np.mean(np.mean((cdf_pred - cdf_obs) ** 2, axis=1)))


def multiclass_log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean cross-entropy loss; clips probabilities to avoid log(0)."""
    yt = np.asarray(y_true)
    yp = np.clip(np.asarray(y_pred, dtype=float), _EPS, 1.0)
    return float(-np.mean(np.log(yp[np.arange(len(yt)), yt])))


def brier_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean multi-class Brier score (mean squared error in probability space)."""
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred, dtype=float)
    one_hot = np.eye(yp.shape[1])[yt]
    return float(np.mean(np.sum((yp - one_hot) ** 2, axis=1)))


# ── Backtest helpers ─────────────────────────────────────────────────────────


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _META_COLS]


def _slice_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    return df[pd.to_datetime(df["date"]).dt.year == year].reset_index(drop=True)


def _pre_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    return df[pd.to_datetime(df["date"]).dt.year < year].reset_index(drop=True)


def _bookmaker_rps(year_matches: pd.DataFrame, odds: pd.DataFrame) -> float | None:
    """Merge bookmaker odds onto year matches and return mean RPS; None if no overlap."""
    key = ["date", "home_team", "away_team"]
    merged = year_matches[key + ["outcome"]].merge(
        odds[key + ["p_win", "p_draw", "p_loss"]], on=key, how="inner"
    )
    if merged.empty:
        return None
    return rank_probability_score(
        merged["outcome"].to_numpy(),
        merged[["p_win", "p_draw", "p_loss"]].to_numpy(),
    )


def _build_row(
    year_matches: pd.DataFrame,
    hybrid: Any,
    legacy: Any,
    feat_cols: list[str],
    odds: pd.DataFrame | None,
) -> dict[str, Any]:
    X = year_matches[feat_cols].to_numpy()
    y_true = year_matches["outcome"].to_numpy()
    row: dict[str, Any] = {
        "rps_hybrid": rank_probability_score(y_true, hybrid.predict_proba(X)),
        "rps_legacy": rank_probability_score(y_true, legacy.predict_proba(X)),
    }
    if odds is not None:
        bk = _bookmaker_rps(year_matches, odds)
        if bk is not None:
            row["rps_bookmaker"] = bk
    return row


def backtest_hybrid(
    matches: pd.DataFrame,
    hybrid: Any,
    legacy: Any,
    odds: pd.DataFrame | None = None,
    years: list[int] | None = None,
) -> pd.DataFrame:
    """Time-aware WC backtest: train on pre-year data, evaluate on WC year.

    Parameters
    ----------
    matches:  DataFrame with 'date', 'outcome', and feature columns.
    hybrid:   Model with fit(X, y) and predict_proba(X) → (N, 3).
    legacy:   Same protocol as hybrid (used as second baseline).
    odds:     Optional bookmaker odds with 'date', 'home_team', 'away_team',
              'p_win', 'p_draw', 'p_loss'. Skipped when None.
    years:    WC years to backtest (default [2014, 2018, 2022]).

    Returns
    -------
    DataFrame indexed by year with columns rps_hybrid, rps_legacy, and
    (when odds are not None) rps_bookmaker.
    """
    eval_years = years if years is not None else _DEFAULT_WC_YEARS
    feat_cols = _feature_cols(matches)
    rows: dict[int, dict[str, Any]] = {}
    for year in eval_years:
        year_matches = _slice_year(matches, year)
        if year_matches.empty:
            continue
        train = _pre_year(matches, year)
        if not train.empty:
            X_tr = train[feat_cols].to_numpy()
            y_tr = train["outcome"].to_numpy()
            hybrid.fit(X_tr, y_tr)
            legacy.fit(X_tr, y_tr)
        rows[year] = _build_row(year_matches, hybrid, legacy, feat_cols, odds)
    result = pd.DataFrame.from_dict(rows, orient="index")
    result.index.name = "year"
    return result


# ── High-level CLI entry ─────────────────────────────────────────────────────


def _read_features(root: Any) -> "pd.DataFrame | None":
    from pathlib import Path
    try:
        df = pd.read_csv(Path(root) / "dataset" / "features.csv")
        return df if "outcome" in df.columns else None
    except (FileNotFoundError, OSError):
        return None


def _sklearn_pair(seed: int = 42) -> tuple[Any, Any]:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    return (
        GradientBoostingClassifier(random_state=seed),
        RandomForestClassifier(n_estimators=10, random_state=seed),
    )


def _build_odds_frame(cfg: Any, odds_mod: Any) -> "pd.DataFrame | None":
    """Load and concat match odds for all configured seasons; None when all empty/blocked."""
    seasons = getattr(getattr(cfg, "odds", None), "seasons", [])
    frames: list[pd.DataFrame] = []
    for year in seasons:
        try:
            raw = odds_mod.load_match_odds(f"wc{year}", cfg)
            if not raw.empty:
                frames.append(raw)
        except Exception:
            logger.warning("Match odds unavailable for wc%d; skipping.", year)
    return pd.concat(frames, ignore_index=True) if frames else None


def run_backtest(cfg: Any = None, root: Any = None) -> "pd.DataFrame | None":
    """High-level CLI entry: load features+targets and run time-aware WC backtest.

    Loads bookmaker match odds for all seasons in cfg.odds.seasons (via
    odds.load_match_odds) and passes them to backtest_hybrid so rps_bookmaker is
    included when odds are available.  Passes odds=None when all scrapers are
    blocked, so backtest_hybrid degrades gracefully.

    Returns None when dataset/features.csv is unavailable or lacks an outcome column.
    """
    from worldcup_playoff.config import AppConfig
    from worldcup_playoff.data import odds as _odds_mod

    resolved = cfg if cfg is not None else AppConfig()
    matches = _read_features(root or ".")
    if matches is None:
        return None
    seed = getattr(getattr(resolved, "hybrid", None), "random_seed", 42)
    combined_odds = _build_odds_frame(resolved, _odds_mod)
    return backtest_hybrid(matches, *_sklearn_pair(seed), odds=combined_odds)


# ── Elo-prior-weight tuning (Dixon-Coles + Elo simulation path) ────────────────

_WC_TOURNAMENT: str = "FIFA World Cup"
_DEFAULT_PRIOR_WEIGHTS: tuple[float, ...] = (0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def _outcome(home_goals: int, away_goals: int) -> int:
    """Ordered 3-class outcome from the home perspective: 0=win, 1=draw, 2=loss."""
    return 0 if home_goals > away_goals else (1 if home_goals == away_goals else 2)


def _wdl_probs(abilities: Any, home: str, away: str, max_goals: int) -> list[float]:
    """Neutral-venue (P_home_win, P_draw, P_away_win) from the Dixon-Coles pmf."""
    from worldcup_playoff.simulation.poisson import lambdas, score_matrix

    lh, la = lambdas(abilities, home, away, neutral=True)
    mat = score_matrix(lh, la, abilities.rho, max_goals)
    hw = float(np.tril(mat, -1).sum())  # home goals > away goals
    aw = float(np.triu(mat, 1).sum())   # away goals > home goals
    return [hw, 1.0 - hw - aw, aw]


def backtest_elo_prior_weight(
    results: pd.DataFrame,
    cfg: Any,
    weights: "list[float] | tuple[float, ...]" = _DEFAULT_PRIOR_WEIGHTS,
    years: list[int] | None = None,
) -> pd.DataFrame:
    """Tune ``poisson.elo_prior_weight`` by match-level RPS over past World Cups.

    For each WC year, fits Dixon-Coles + Elo on all matches **before** that
    tournament (so there is no leakage), then for each candidate weight blends
    the abilities and scores every WC match's neutral-venue W/D/L probability
    against the actual result. Returns a DataFrame indexed by ``weight`` with
    pooled ``rps`` / ``log_loss`` / ``brier`` plus per-year ``rps_<year>``
    columns. Lower RPS is better.

    Expects the coerced martj42 schema (``DATE``, ``HOME_TEAM``, ``AWAY_TEAM``,
    ``HOME_GOALS``, ``AWAY_GOALS``, ``TOURNAMENT``).
    """
    from worldcup_playoff.data.elo import compute_elo
    from worldcup_playoff.simulation.poisson import blend_abilities_with_elo, fit_dixon_coles

    target_years = years or _DEFAULT_WC_YEARS
    max_goals = cfg.poisson.max_goals
    df = results.copy()
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")

    # Fit Dixon-Coles + Elo once per year (independent of the blend weight).
    fitted: dict[int, tuple[Any, dict[str, float], list[tuple[str, str, int]]]] = {}
    for year in target_years:
        train = df[df["DATE"] < pd.Timestamp(f"{year}-05-01")]
        wc = df[
            (df["TOURNAMENT"] == _WC_TOURNAMENT)
            & (df["DATE"].dt.year == year)
            & df["HOME_GOALS"].notna()
            & df["AWAY_GOALS"].notna()
        ]
        if train.empty or wc.empty:
            continue
        abilities = fit_dixon_coles(train, cfg.poisson)
        elo = compute_elo(train, getattr(cfg, "elo", None)).final_ratings
        matches = [
            (r.HOME_TEAM, r.AWAY_TEAM, _outcome(int(r.HOME_GOALS), int(r.AWAY_GOALS)))
            for r in wc.itertuples(index=False)
            if r.HOME_TEAM in abilities.attack and r.AWAY_TEAM in abilities.attack
        ]
        if matches:
            fitted[year] = (abilities, elo, matches)

    rows: list[dict[str, float]] = []
    for weight in weights:
        per_year: dict[int, float] = {}
        y_true_all: list[int] = []
        y_pred_all: list[list[float]] = []
        for year, (abilities, elo, matches) in fitted.items():
            blended = blend_abilities_with_elo(abilities, elo, weight)
            yt = [o for _, _, o in matches]
            yp = [_wdl_probs(blended, h, a, max_goals) for h, a, _ in matches]
            per_year[year] = rank_probability_score(np.array(yt), np.array(yp))
            y_true_all.extend(yt)
            y_pred_all.extend(yp)
        if not y_true_all:
            continue
        yt_arr, yp_arr = np.array(y_true_all), np.array(y_pred_all)
        row: dict[str, float] = {
            "weight": float(weight),
            "rps": rank_probability_score(yt_arr, yp_arr),
            "log_loss": multiclass_log_loss(yt_arr, yp_arr),
            "brier": brier_score(yt_arr, yp_arr),
        }
        row.update({f"rps_{year}": r for year, r in per_year.items()})
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["rps", "log_loss", "brier"]).rename_axis("weight")
    return pd.DataFrame(rows).set_index("weight")


def run_prior_tuning(
    cfg: Any = None,
    root: Any = None,
    weights: "list[float] | tuple[float, ...] | None" = None,
) -> "pd.DataFrame | None":
    """High-level CLI entry: load martj42 results and tune ``elo_prior_weight``.

    Returns a weight-indexed DataFrame of RPS / log-loss / Brier, or None when
    the martj42 results cache is unavailable. ``root`` is accepted for signature
    parity with :func:`run_backtest` (martj42 paths come from ``cfg.martj42``).
    """
    from worldcup_playoff.config import AppConfig
    from worldcup_playoff.data.martj42_loader import load_martj42_results

    resolved = cfg if cfg is not None else AppConfig()
    try:
        results = load_martj42_results(resolved.martj42)
    except Exception:
        return None
    if results is None or results.empty:
        return None
    return backtest_elo_prior_weight(
        results, resolved, weights=weights or _DEFAULT_PRIOR_WEIGHTS
    )


# ── Market-value-prior tuning (validated on the WC2026 group stage) ────────────

_DEFAULT_MV_WEIGHTS: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
_WC2026_GROUP_START: pd.Timestamp = pd.Timestamp("2026-06-11")
_WC2026_KNOCKOUT_START: pd.Timestamp = pd.Timestamp("2026-06-28")


def backtest_market_value_weight(
    results: pd.DataFrame,
    cfg: Any,
    squad_values: dict[str, float],
    weights: "list[float] | tuple[float, ...]" = _DEFAULT_MV_WEIGHTS,
    elo_weight: float | None = None,
) -> pd.DataFrame:
    """Tune ``poisson.market_value_prior_weight`` on the WC2026 group stage.

    Historical squad values for past World Cups are not bundled, so the market-
    value prior is validated on the only data we have values for: the real 2026
    group stage. Dixon-Coles + Elo are fit on all matches **before** the
    tournament (train cutoff = the first group match, so there is no leakage);
    abilities are blended with Elo at ``elo_weight`` (default
    ``cfg.poisson.elo_prior_weight``) and then, for each candidate weight, with
    the squad-market-value prior. Each played group match is scored by its
    neutral-venue W/D/L probability against the real result.

    Returns a ``market_value_weight``-indexed DataFrame of pooled ``rps`` /
    ``log_loss`` / ``brier`` (+ ``n_matches``). Lower RPS is better.
    """
    from worldcup_playoff.data.elo import compute_elo
    from worldcup_playoff.simulation.poisson import (
        blend_abilities_with_elo,
        blend_abilities_with_market_value,
        fit_dixon_coles,
    )

    df = results.copy()
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    train = df[df["DATE"] < _WC2026_GROUP_START]
    grp = df[
        (df["TOURNAMENT"] == _WC_TOURNAMENT)
        & (df["DATE"] >= _WC2026_GROUP_START)
        & (df["DATE"] < _WC2026_KNOCKOUT_START)
        & df["HOME_GOALS"].notna()
        & df["AWAY_GOALS"].notna()
    ]
    empty = pd.DataFrame(
        columns=["rps", "log_loss", "brier", "n_matches"]
    ).rename_axis("market_value_weight")
    if train.empty or grp.empty:
        return empty

    abilities = fit_dixon_coles(train, cfg.poisson)
    ew = elo_weight if elo_weight is not None else getattr(cfg.poisson, "elo_prior_weight", 0.0)
    if ew > 0.0:
        abilities = blend_abilities_with_elo(
            abilities, compute_elo(train, getattr(cfg, "elo", None)).final_ratings, ew
        )
    matches = [
        (r.HOME_TEAM, r.AWAY_TEAM, _outcome(int(r.HOME_GOALS), int(r.AWAY_GOALS)))
        for r in grp.itertuples(index=False)
        if r.HOME_TEAM in abilities.attack and r.AWAY_TEAM in abilities.attack
    ]
    if not matches:
        return empty

    max_goals = cfg.poisson.max_goals
    yt = np.array([o for _, _, o in matches])
    rows: list[dict[str, float]] = []
    for w in weights:
        blended = blend_abilities_with_market_value(abilities, squad_values, w)
        yp = np.array([_wdl_probs(blended, h, a, max_goals) for h, a, _ in matches])
        rows.append(
            {
                "market_value_weight": float(w),
                "rps": rank_probability_score(yt, yp),
                "log_loss": multiclass_log_loss(yt, yp),
                "brier": brier_score(yt, yp),
                "n_matches": float(len(matches)),
            }
        )
    return pd.DataFrame(rows).set_index("market_value_weight")


def run_market_value_tuning(
    cfg: Any = None,
    root: Any = None,
    weights: "list[float] | tuple[float, ...] | None" = None,
) -> "pd.DataFrame | None":
    """High-level CLI entry: tune ``market_value_prior_weight`` on the 2026 groups.

    Returns a weight-indexed DataFrame of RPS / log-loss / Brier, or None when the
    martj42 cache is unavailable. ``root`` is accepted for parity with the other
    tuning entries.
    """
    from worldcup_playoff.config import AppConfig
    from worldcup_playoff.data.martj42_loader import load_martj42_results
    from worldcup_playoff.data.squad_value import WC2026_SQUAD_VALUE_EUR_M

    resolved = cfg if cfg is not None else AppConfig()
    try:
        results = load_martj42_results(resolved.martj42)
    except Exception:
        return None
    if results is None or results.empty:
        return None
    return backtest_market_value_weight(
        results, resolved, WC2026_SQUAD_VALUE_EUR_M, weights or _DEFAULT_MV_WEIGHTS
    )


# ── 2D prior tuning (elo × market value) over past World Cups ──────────────────

_DEFAULT_2D_ELO_WEIGHTS: tuple[float, ...] = (0.0, 0.4, 0.8)
_DEFAULT_2D_MV_WEIGHTS: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8)
_HISTORICAL_WC_YEARS: tuple[int, ...] = (2018, 2022)


def backtest_2d_prior_weights(
    results: pd.DataFrame,
    cfg: Any,
    squad_values_by_year: dict[int, dict[str, float]],
    elo_weights: "list[float] | tuple[float, ...]" = _DEFAULT_2D_ELO_WEIGHTS,
    mv_weights: "list[float] | tuple[float, ...]" = _DEFAULT_2D_MV_WEIGHTS,
    years: "list[int] | tuple[int, ...]" = _HISTORICAL_WC_YEARS,
) -> pd.DataFrame:
    """Joint (Elo × market-value) prior tuning, pooled over past World Cups.

    For each year with bundled squad values, fits Dixon-Coles + Elo on all
    matches **before** the tournament (no leakage), then scores that year's WC
    matches for every ``(elo_weight, market_value_weight)`` grid point (Elo blend
    first, then the as-of-tournament market-value blend). Returns a long DataFrame
    with columns ``elo_weight``, ``market_value_weight``, ``rps``, ``log_loss``,
    ``brier``, ``n_matches``. Lower RPS is better.
    """
    from worldcup_playoff.data.elo import compute_elo
    from worldcup_playoff.simulation.poisson import (
        blend_abilities_with_elo,
        blend_abilities_with_market_value,
        fit_dixon_coles,
    )

    df = results.copy()
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    max_goals = cfg.poisson.max_goals

    fitted: dict[int, tuple[Any, dict[str, float], list[tuple[str, str, int]], dict[str, float]]] = {}
    for year in years:
        sv = squad_values_by_year.get(year)
        if not sv:
            continue
        train = df[df["DATE"] < pd.Timestamp(f"{year}-05-01")]
        wc = df[
            (df["TOURNAMENT"] == _WC_TOURNAMENT)
            & (df["DATE"].dt.year == year)
            & df["HOME_GOALS"].notna()
            & df["AWAY_GOALS"].notna()
        ]
        if train.empty or wc.empty:
            continue
        abilities = fit_dixon_coles(train, cfg.poisson)
        elo = compute_elo(train, getattr(cfg, "elo", None)).final_ratings
        matches = [
            (r.HOME_TEAM, r.AWAY_TEAM, _outcome(int(r.HOME_GOALS), int(r.AWAY_GOALS)))
            for r in wc.itertuples(index=False)
            if r.HOME_TEAM in abilities.attack and r.AWAY_TEAM in abilities.attack
        ]
        if matches:
            fitted[year] = (abilities, elo, matches, sv)

    rows: list[dict[str, float]] = []
    for ew in elo_weights:
        for mw in mv_weights:
            yt_all: list[int] = []
            yp_all: list[list[float]] = []
            for _year, (abilities, elo, matches, sv) in fitted.items():
                blended = blend_abilities_with_elo(abilities, elo, ew) if ew > 0 else abilities
                if mw > 0:
                    blended = blend_abilities_with_market_value(blended, sv, mw)
                yt_all.extend(o for _, _, o in matches)
                yp_all.extend(_wdl_probs(blended, h, a, max_goals) for h, a, _ in matches)
            if not yt_all:
                continue
            yt, yp = np.array(yt_all), np.array(yp_all)
            rows.append(
                {
                    "elo_weight": float(ew),
                    "market_value_weight": float(mw),
                    "rps": rank_probability_score(yt, yp),
                    "log_loss": multiclass_log_loss(yt, yp),
                    "brier": brier_score(yt, yp),
                    "n_matches": float(len(yt_all)),
                }
            )
    return pd.DataFrame(rows)


def run_2d_prior_tuning(cfg: Any = None, root: Any = None) -> "pd.DataFrame | None":
    """High-level CLI entry: joint Elo × market-value tuning over WC2018/2022."""
    from worldcup_playoff.config import AppConfig
    from worldcup_playoff.data.martj42_loader import load_martj42_results
    from worldcup_playoff.data.squad_value import WC_SQUAD_VALUE_HISTORICAL_EUR_M

    resolved = cfg if cfg is not None else AppConfig()
    try:
        results = load_martj42_results(resolved.martj42)
    except Exception:
        return None
    if results is None or results.empty:
        return None
    return backtest_2d_prior_weights(results, resolved, WC_SQUAD_VALUE_HISTORICAL_EUR_M)


class ModelEvaluator:
    """Evaluates classifier performance."""

    @staticmethod
    def evaluate(
        classifier: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict[str, Any]:
        """Return confusion matrix and classification report as a dict."""
        y_pred = classifier.predict(X_test)
        return {
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "classification_report": classification_report(
                y_test, y_pred, output_dict=True
            ),
        }

    @staticmethod
    def plot_roc_curves(
        classifiers: dict[str, Any],
        X_test: np.ndarray,
        y_test: np.ndarray,
        output_path: Path | None = None,
    ) -> None:
        """Plot ROC curves for multiple classifiers."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 8))
        linestyles = ["-", ":", "--", "-.", (0, (3, 1, 1, 1))]

        for i, (name, clf) in enumerate(classifiers.items()):
            RocCurveDisplay.from_estimator(
                clf, X_test, y_test,
                ax=ax,
                linewidth=3,
                linestyle=linestyles[i % len(linestyles)],
                name=name,
            )

        ax.tick_params(axis="both", labelsize=16)
        ax.set_xlabel("False Positive Rate", fontsize=18)
        ax.set_ylabel("True Positive Rate", fontsize=18)
        ax.legend(fontsize=14)

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, bbox_inches="tight")
            logger.info("ROC curve saved to %s", output_path)
        plt.close(fig)
