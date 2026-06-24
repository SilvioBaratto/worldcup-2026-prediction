"""Archived bookmaker odds scraper, de-vig, and CSV cache (backtest baseline only)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup

import worldcup_playoff.data.crosswalk as _cw
from worldcup_playoff.config import OddsConfig

logger = logging.getLogger(__name__)

_SEASON_URLS: dict[str, str] = {
    "wc2014": "https://www.oddsportal.com/soccer/world/world-cup-2014/results/",
    "wc2018": "https://www.oddsportal.com/soccer/world/world-cup-2018/results/",
    "wc2022": "https://www.oddsportal.com/soccer/world/world-cup-2022/results/",
}

# Match 1X2 odds — parallel cache path avoids collision with outright ({tournament}_odds.csv).
_MATCH_SEASON_URLS: dict[str, str] = {
    "wc2014": "https://www.oddsportal.com/soccer/world/world-cup-2014/results/",
    "wc2018": "https://www.oddsportal.com/soccer/world/world-cup-2018/results/",
    "wc2022": "https://www.oddsportal.com/soccer/world/world-cup-2022/results/",
}

_MATCH_ODDS_COLS = ["date", "home_team", "away_team", "o_home", "o_draw", "o_away"]
_MATCH_PROB_COLS = ["date", "home_team", "away_team", "p_win", "p_draw", "p_loss"]


def de_vig(odds: list[float]) -> list[float]:
    """De-vig decimal odds to probabilities: p_i = (1/o_i) / Σ(1/o_j)."""
    inv = [1.0 / o for o in odds]
    total = sum(inv)
    return [v / total for v in inv]


# ── Match odds (1X2) ─────────────────────────────────────────────────────────


def _fetch_match_html(url: str) -> str:
    """Fetch raw HTML from *url*. Replace at module level in tests to avoid live calls."""
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return str(resp.text)


def _parse_match_row(row: Any) -> dict[str, Any] | None:
    """Extract date/teams/1X2 odds from one table row; return None on any parse failure."""
    date_tag = row.select_one(".table-time")
    home_tag = row.select_one(".table-home")
    away_tag = row.select_one(".table-away")
    o_home_tag = row.select_one(".odds-home")
    o_draw_tag = row.select_one(".odds-draw")
    o_away_tag = row.select_one(".odds-away")
    if not all([date_tag, home_tag, away_tag, o_home_tag, o_draw_tag, o_away_tag]):
        return None
    try:
        return {
            "date": date_tag.get_text(strip=True),
            "home_team": home_tag.get_text(strip=True),
            "away_team": away_tag.get_text(strip=True),
            "o_home": float(o_home_tag.get_text(strip=True)),
            "o_draw": float(o_draw_tag.get_text(strip=True)),
            "o_away": float(o_away_tag.get_text(strip=True)),
        }
    except (ValueError, AttributeError):
        return None


def parse_match_html(html: str) -> pd.DataFrame:
    """Parse an archived match-odds page; return date/home_team/away_team/o_home/o_draw/o_away."""
    soup = BeautifulSoup(html, "html.parser")
    rows = [r for row in soup.select("tr") for r in [_parse_match_row(row)] if r]
    if not rows:
        return pd.DataFrame(columns=_MATCH_ODDS_COLS)
    return pd.DataFrame(rows)


def to_match_probs(df: pd.DataFrame) -> pd.DataFrame:
    """De-vig each row's 1X2 odds to WDL probabilities (p_win, p_draw, p_loss)."""
    inv = df[["o_home", "o_draw", "o_away"]].rdiv(1.0)
    totals = inv.sum(axis=1)
    return df.assign(
        p_win=inv["o_home"] / totals,
        p_draw=inv["o_draw"] / totals,
        p_loss=inv["o_away"] / totals,
    )


def _fetch_and_cache_match(tournament: str, path: Path) -> pd.DataFrame:
    try:
        url = _MATCH_SEASON_URLS.get(tournament, "")
        html = _fetch_match_html(url)
        raw = parse_match_html(html)
        raw["home_team"] = raw["home_team"].map(_cw.normalize_team)
        raw["away_team"] = raw["away_team"].map(_cw.normalize_team)
        result = to_match_probs(raw)
        path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(path, index=False)
        return result
    except Exception:
        logger.warning("Match odds fetch failed for %s; returning empty frame.", tournament)
        return pd.DataFrame(columns=_MATCH_PROB_COLS)


def load_match_odds(tournament: str, config: Any) -> pd.DataFrame:
    """Cache-first loader for match 1X2 odds; degrades to empty frame on block, never raises."""
    cache_dir = Path(config.odds.cache_dir)
    path = cache_dir / f"{tournament}_match_odds.csv"
    if path.exists():
        return pd.read_csv(path)
    return _fetch_and_cache_match(tournament, path)


# ── Outright champion odds ────────────────────────────────────────────────────


class OddsScraper:
    """Cache-first scraper for WC2014/18/22 bookmaker odds (backtest baseline only)."""

    def __init__(
        self,
        config: OddsConfig | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        cfg = config or OddsConfig()
        self._config = cfg
        self._cache_dir = cache_dir if cache_dir is not None else cfg.cache_dir

    def load(self, tournament: str) -> pd.DataFrame:
        """Return the odds DataFrame for *tournament*, reading from cache when available."""
        path = self._cache_path(tournament)
        if path.exists():
            return pd.read_csv(path)
        return self._fetch_and_cache(tournament, path)

    def parse_html(self, html: str) -> pd.DataFrame:
        """Parse a bookmaker odds page with BeautifulSoup; return 'team'/'odds' DataFrame."""
        soup = BeautifulSoup(html, "html.parser")
        rows = [r for row in soup.select("tr") for r in [self._parse_row(row)] if r]
        if not rows:
            return pd.DataFrame(columns=["team", "odds"])
        return pd.DataFrame(rows, columns=["team", "odds"])

    # ── private ──────────────────────────────────────────────────────────────

    def _cache_path(self, tournament: str) -> Path:
        return self._cache_dir / f"{tournament}_odds.csv"

    def _fetch_and_cache(self, tournament: str, path: Path) -> pd.DataFrame:
        try:
            url = _SEASON_URLS.get(tournament, self._config.source_url)
            html = self._fetch_html(url)
            df = self.parse_html(html)
            df["team"] = df["team"].map(_cw.normalize_team)
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False)
            return df
        except Exception:
            logger.warning("Odds fetch failed for %s; returning empty frame.", tournament)
            return pd.DataFrame(columns=["team", "odds"])

    def _fetch_html(self, url: str) -> str:
        """Fetch raw HTML from *url*. Assign to the instance to override in tests."""
        headers = {"User-Agent": self._config.user_agent}
        resp = requests.get(url, headers=headers, timeout=self._config.request_timeout)
        resp.raise_for_status()
        return str(resp.text)

    def _parse_row(self, row: Any) -> tuple[str, float] | None:
        team_tag = row.select_one(".team-name") or row.select_one("td.table-participant")
        odds_tag = row.select_one(".table-odds")
        if not (team_tag and odds_tag):
            return None
        try:
            return (team_tag.get_text(strip=True), float(odds_tag.get_text(strip=True)))
        except (ValueError, AttributeError):
            return None


def load_odds(tournament: str, config: OddsConfig | None = None) -> pd.DataFrame:
    """Module-level convenience: load archived odds for *tournament*."""
    return OddsScraper(config=config).load(tournament)
