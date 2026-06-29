"""Dixon-Coles bivariate-Poisson estimator with exponential time-decay.

Fits per-team attack/defence abilities (plus home advantage, rho, intercept) via
weighted MLE on martj42 match history, then provides a deterministic scoreline
sampler that draws from the τ-corrected joint Poisson pmf grid.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson as sp_poisson

from worldcup_playoff.config import PoissonConfig

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Value objects
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TeamAbilities:
    """Fitted per-team attack/defence abilities from the Dixon-Coles model."""

    attack: dict[str, float]
    defence: dict[str, float]
    home_adv: float
    rho: float
    intercept: float


def blend_abilities_with_elo(
    abilities: TeamAbilities,
    elo_ratings: dict[str, float],
    weight: float,
) -> TeamAbilities:
    """Shrink Dixon-Coles attack/defence toward an Elo-implied strength prior.

    The fitted abilities are MLE estimates on raw goals, so teams that ran up
    large scorelines (e.g. in the current group stage) get extreme abilities
    while reputable sides with modest recent margins are understated. Blending
    toward an Elo prior re-injects long-run team strength.

    For every team present in both ``abilities`` and ``elo_ratings``, the Elo
    rating is standardized across that shared set and rescaled to the spread of
    the fitted abilities, yielding a prior on the same log-rate scale. Each
    team's attack and defence are then blended ``(1 - weight) * fitted + weight
    * prior``. ``weight <= 0`` returns the abilities unchanged; teams without an
    Elo rating are never touched. The prior is balanced (one Elo z-score shifts
    attack and defence equally), so it nudges overall quality without inventing
    an attack/defence skew. ``home_adv``, ``rho`` and ``intercept`` are kept.
    """
    if weight <= 0.0:
        return abilities
    shared = [t for t in abilities.attack if t in elo_ratings]
    if len(shared) < 2:
        return abilities
    elos = np.array([elo_ratings[t] for t in shared], dtype=float)
    elo_mean, elo_std = float(elos.mean()), float(elos.std())
    if elo_std == 0.0:
        return abilities

    atk = np.array([abilities.attack[t] for t in shared], dtype=float)
    dfc = np.array([abilities.defence[t] for t in shared], dtype=float)
    atk_mean, atk_std = float(atk.mean()), float(atk.std())
    dfc_mean, dfc_std = float(dfc.mean()), float(dfc.std())

    new_attack = dict(abilities.attack)
    new_defence = dict(abilities.defence)
    for team in shared:
        z = (elo_ratings[team] - elo_mean) / elo_std
        attack_prior = atk_mean + z * atk_std
        defence_prior = dfc_mean + z * dfc_std
        new_attack[team] = (1.0 - weight) * abilities.attack[team] + weight * attack_prior
        new_defence[team] = (1.0 - weight) * abilities.defence[team] + weight * defence_prior

    return TeamAbilities(
        attack=new_attack,
        defence=new_defence,
        home_adv=abilities.home_adv,
        rho=abilities.rho,
        intercept=abilities.intercept,
    )


def blend_abilities_with_market_value(
    abilities: TeamAbilities,
    squad_values_eur_m: dict[str, float],
    weight: float,
) -> TeamAbilities:
    """Shrink abilities toward a squad-market-value strength prior (player-level signal).

    Market value is a "wisdom of the crowd" estimate of the *current* squad
    quality — information a results-only history (Elo / fitted goals) misses. We
    take ``log(value)`` (squad values span ~20m to ~1.5bn, a heavy right tail) so
    the prior is roughly linear in strength, then reuse the same standardize-and-
    blend logic as the Elo prior. ``weight <= 0`` returns the abilities unchanged;
    teams without a value are left untouched. Apply *after* the Elo blend to add
    the squad signal on top of the long-run strength prior.
    """
    if weight <= 0.0:
        return abilities
    log_values = {
        team: float(np.log(value))
        for team, value in squad_values_eur_m.items()
        if value > 0 and team in abilities.attack
    }
    return blend_abilities_with_elo(abilities, log_values, weight)


@dataclass(frozen=True)
class _PreparedData:
    teams: list[str]
    home_idx: np.ndarray
    away_idx: np.ndarray
    h_goals: np.ndarray
    a_goals: np.ndarray
    neutral_mask: np.ndarray
    weights: np.ndarray


# ─────────────────────────────────────────────────────────────────────────────
# Pure functions
# ─────────────────────────────────────────────────────────────────────────────


def dixon_coles_tau(h: int, a: int, lh: float, la: float, rho: float) -> float:
    """Dixon-Coles τ correction factor for low-score cells (h+a ≤ 1 or h==a==1)."""
    if h == 0 and a == 0:
        return 1.0 - lh * la * rho
    if h == 1 and a == 0:
        return 1.0 + la * rho
    if h == 0 and a == 1:
        return 1.0 + lh * rho
    if h == 1 and a == 1:
        return 1.0 - rho
    return 1.0


def decay_weight(age_days: float | np.ndarray, half_life_days: float) -> float | np.ndarray:
    """Exponential decay: weight = 0.5^(age_days / half_life_days)."""
    return 0.5 ** (age_days / half_life_days)


def lambdas(
    abilities: TeamAbilities,
    home: str,
    away: str,
    neutral: bool = False,
) -> tuple[float, float]:
    """Compute (λ_home, λ_away) expected goal rates for one match."""
    ha = 0.0 if neutral else abilities.home_adv
    lh = float(np.exp(abilities.intercept + abilities.attack[home] - abilities.defence[away] + ha))
    la = float(np.exp(abilities.intercept + abilities.attack[away] - abilities.defence[home]))
    return lh, la


def score_matrix(
    lh: float,
    la: float,
    rho: float = 0.0,
    max_goals: int = 10,
) -> np.ndarray:
    """Normalized (max_goals+1)×(max_goals+1) joint pmf grid with τ correction."""
    goals = np.arange(max_goals + 1)
    mat = np.outer(sp_poisson.pmf(goals, lh), sp_poisson.pmf(goals, la))
    _apply_tau_correction(mat, lh, la, rho, max_goals)
    return mat / mat.sum()  # type: ignore[no-any-return]


def modal_scoreline(
    abilities: TeamAbilities,
    home: str,
    away: str,
    max_goals: int = 10,
    neutral: bool = True,
) -> tuple[int, int]:
    """Most-likely (home_goals, away_goals) — the argmax of the Dixon-Coles score matrix."""
    lh, la = lambdas(abilities, home, away, neutral=neutral)
    mat = score_matrix(lh, la, rho=abilities.rho, max_goals=max_goals)
    flat = int(mat.argmax())
    cols = mat.shape[1]
    return flat // cols, flat % cols


def decisive_scoreline(
    abilities: TeamAbilities,
    home: str,
    away: str,
    home_wins: bool,
    max_goals: int = 10,
    neutral: bool = True,
) -> tuple[int, int]:
    """Most-likely non-draw (home_goals, away_goals) with the given winner.

    Argmax over the decisive half of the Dixon-Coles score matrix (home > away,
    or away > home). Used for knockout ties, where a draw is settled by extra time
    or penalties — so a drawn "predicted score" would contradict the team shown
    advancing. Falls back to the favourite winning by one goal in the unreachable
    edge case where the decisive half is empty.
    """
    lh, la = lambdas(abilities, home, away, neutral=neutral)
    mat = score_matrix(lh, la, rho=abilities.rho, max_goals=max_goals)
    rows = np.arange(mat.shape[0])[:, None]
    cols_idx = np.arange(mat.shape[1])[None, :]
    decisive = (rows > cols_idx) if home_wins else (rows < cols_idx)
    if not decisive.any():
        return (1, 0) if home_wins else (0, 1)
    masked = np.where(decisive, mat, -1.0)
    flat = int(masked.argmax())
    cols = mat.shape[1]
    return flat // cols, flat % cols


def _apply_tau_correction(
    mat: np.ndarray, lh: float, la: float, rho: float, max_goals: int
) -> None:
    """Apply τ correction in-place for the four low-score cells, clipping to 0."""
    for h in range(min(2, max_goals + 1)):
        for a in range(min(2, max_goals + 1)):
            mat[h, a] *= dixon_coles_tau(h, a, lh, la, rho)
    np.maximum(mat, 0.0, out=mat)  # τ can go negative for large |rho|; clip to valid pmf


# ─────────────────────────────────────────────────────────────────────────────
# Preparation helpers (module-level to keep DixonColesEstimator small)
# ─────────────────────────────────────────────────────────────────────────────


def _filter_played(df: pd.DataFrame) -> pd.DataFrame:
    mask = ~df["HOME_GOALS"].isna() & ~df["AWAY_GOALS"].isna()
    played = df[mask].copy()
    played["_date"] = pd.to_datetime(played["DATE"])
    return played


def _compute_weights(played: pd.DataFrame, half_life_days: float) -> np.ndarray:
    ref = played["_date"].max()
    age = (ref - played["_date"]).dt.days.astype(float)
    return np.asarray(decay_weight(age.values, half_life_days))


def _build_team_index(played: pd.DataFrame) -> tuple[list[str], np.ndarray, np.ndarray]:
    teams = sorted(set(played["HOME_TEAM"]) | set(played["AWAY_TEAM"]))
    idx = {t: i for i, t in enumerate(teams)}
    return teams, played["HOME_TEAM"].map(idx).values, played["AWAY_TEAM"].map(idx).values


def _extract_neutral(played: pd.DataFrame) -> np.ndarray:
    if "NEUTRAL" in played.columns:
        return played["NEUTRAL"].values.astype(bool)  # type: ignore[no-any-return]
    return np.zeros(len(played), dtype=bool)


def _prepare(df: pd.DataFrame, half_life_days: float) -> _PreparedData:
    """Drop unplayed rows, compute decay weights, build team index."""
    played = _filter_played(df)
    weights = _compute_weights(played, half_life_days)
    teams, h_idx, a_idx = _build_team_index(played)
    hg = played["HOME_GOALS"].astype(int).values
    ag = played["AWAY_GOALS"].astype(int).values
    return _PreparedData(teams, h_idx, a_idx, hg, ag, _extract_neutral(played), weights)


# ─────────────────────────────────────────────────────────────────────────────
# Negative log-likelihood (module-level; passed to scipy.optimize.minimize)
# ─────────────────────────────────────────────────────────────────────────────


def _tau_vec(
    h: np.ndarray,
    a: np.ndarray,
    lh: np.ndarray,
    la: np.ndarray,
    rho: float,
) -> np.ndarray:
    """Vectorized τ corrections for arrays of match scores."""
    tau = np.ones(len(h))
    m00, m10, m01, m11 = (h == 0) & (a == 0), (h == 1) & (a == 0), (h == 0) & (a == 1), (h == 1) & (a == 1)
    tau[m00] = 1.0 - lh[m00] * la[m00] * rho
    tau[m10] = 1.0 + la[m10] * rho
    tau[m01] = 1.0 + lh[m01] * rho
    tau[m11] = 1.0 - rho
    return tau


def _nll(params: np.ndarray, data: _PreparedData) -> float:
    """Negative weighted DC log-likelihood; τ clipped to ε for numerical stability."""
    n = len(data.teams)
    atk, dfn = params[:n], params[n : 2 * n]
    home_adv, rho, intercept = params[2 * n], params[2 * n + 1], params[2 * n + 2]
    ha = np.where(data.neutral_mask, 0.0, home_adv)
    lh = np.exp(intercept + atk[data.home_idx] - dfn[data.away_idx] + ha)
    la = np.exp(intercept + atk[data.away_idx] - dfn[data.home_idx])
    tau = _tau_vec(data.h_goals, data.a_goals, lh, la, rho)
    ll = sp_poisson.logpmf(data.h_goals, lh) + sp_poisson.logpmf(data.a_goals, la)
    ll += np.log(np.maximum(tau, 1e-10))
    return float(-(data.weights * ll).sum())


def _normalize_params(params: np.ndarray, n: int) -> np.ndarray:
    """Mean-zero attack and defence; compensate intercept so fitted λ are unchanged.

    λ = exp(intercept + attack_h − defence_a + ha), so shifting both by their means
    requires intercept' = intercept + mean_atk − mean_dfn.
    """
    result = params.copy()
    mean_atk = float(np.mean(result[:n]))
    mean_dfn = float(np.mean(result[n : 2 * n]))
    result[:n] -= mean_atk
    result[n : 2 * n] -= mean_dfn
    result[2 * n + 2] += mean_atk - mean_dfn
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Estimator
# ─────────────────────────────────────────────────────────────────────────────


class DixonColesEstimator:
    """Fits attack/defence abilities via weighted MLE (L-BFGS-B)."""

    def __init__(
        self,
        config: PoissonConfig | None = None,
        *,
        half_life_days: float | None = None,
    ) -> None:
        cfg = config or PoissonConfig()
        if half_life_days is not None:
            cfg = cfg.model_copy(update={"half_life_days": half_life_days})
        self._cfg = cfg

    def fit(self, df: pd.DataFrame) -> TeamAbilities:
        """Return TeamAbilities fitted on played rows of *df*."""
        data = _prepare(df, self._cfg.half_life_days)
        n = len(data.teams)
        x0, bounds = self._initial_params(n)
        res = minimize(_nll, x0, args=(data,), method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": self._cfg.optimizer_maxiter})
        return self._unpack(res.x, data.teams)

    def _initial_params(self, n: int) -> tuple[np.ndarray, list[tuple[float | None, float | None]]]:
        x0 = np.zeros(2 * n + 3)
        x0[2 * n] = self._cfg.home_adv_init
        x0[2 * n + 1] = self._cfg.rho_init
        bounds: list[tuple[float | None, float | None]] = []
        bounds.extend([(None, None)] * (2 * n))
        bounds.extend([(None, None), (-0.99, 0.0), (None, None)])
        return x0, bounds

    def _unpack(self, x: np.ndarray, teams: list[str]) -> TeamAbilities:
        n = len(teams)
        normed = _normalize_params(x, n)
        atk = {t: float(normed[i]) for i, t in enumerate(teams)}
        dfn = {t: float(normed[n + i]) for i, t in enumerate(teams)}
        return TeamAbilities(atk, dfn, float(normed[2 * n]), float(normed[2 * n + 1]), float(normed[2 * n + 2]))


# ─────────────────────────────────────────────────────────────────────────────
# Scoreline sampler
# ─────────────────────────────────────────────────────────────────────────────


class ScorelineSampler:
    """Draws scorelines from the τ-corrected joint Poisson pmf."""

    def __init__(self, abilities: TeamAbilities, config: PoissonConfig | None = None) -> None:
        self._abilities = abilities
        self._cfg = config or PoissonConfig()

    def __call__(
        self,
        home: str,
        away: str,
        rng: np.random.Generator,
        neutral: bool = False,
    ) -> tuple[int, int]:
        """Draw one (home_goals, away_goals) pair advancing the injected Generator."""
        lh, la = lambdas(self._abilities, home, away, neutral=neutral)
        mat = score_matrix(lh, la, rho=self._abilities.rho, max_goals=self._cfg.max_goals)
        g = self._cfg.max_goals + 1
        idx = int(rng.choice(g * g, p=mat.ravel()))
        return idx // g, idx % g

    def sample(
        self,
        home: str,
        away: str,
        neutral: bool = False,
        size: int = 1,
        random_state: int | None = None,
    ) -> np.ndarray:
        """Return int array of shape (size, 2): [home_goals, away_goals] per row."""
        seed = random_state if random_state is not None else self._cfg.random_seed
        rng = np.random.default_rng(seed)
        lh, la = lambdas(self._abilities, home, away, neutral=neutral)
        mat = score_matrix(lh, la, rho=self._abilities.rho, max_goals=self._cfg.max_goals)
        g = self._cfg.max_goals + 1
        choices = rng.choice(g * g, size=size, p=mat.ravel())
        return np.stack([choices // g, choices % g], axis=1).astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# Factories
# ─────────────────────────────────────────────────────────────────────────────


def fit_dixon_coles(df: pd.DataFrame, config: PoissonConfig | None = None) -> TeamAbilities:
    """Fit and return Dixon-Coles abilities from a results DataFrame."""
    return DixonColesEstimator(config).fit(df)


def _dict_to_team_abilities(
    d: dict[str, dict[str, float]],
    config: PoissonConfig | None,
) -> TeamAbilities:
    """Convert ``{team: {attack, defence}}`` dict to TeamAbilities (home_adv/rho from config)."""
    attack = {team: float(vals["attack"]) for team, vals in d.items()}
    defence = {team: float(vals["defence"]) for team, vals in d.items()}
    if config is not None:
        home_adv = config.home_adv_init
        rho = config.rho_init
    else:
        home_adv = 0.25
        rho = -0.1
    return TeamAbilities(attack=attack, defence=defence, home_adv=home_adv, rho=rho, intercept=0.0)


def make_sampler(
    abilities: TeamAbilities | dict[str, dict[str, float]],
    config: PoissonConfig | None = None,
) -> ScorelineSampler:
    """Create a ScorelineSampler from fitted abilities or a raw ``{team: {attack, defence}}`` dict."""
    if isinstance(abilities, dict):
        abilities = _dict_to_team_abilities(abilities, config)
    return ScorelineSampler(abilities, config)
