"""Archived bookmaker odds scraper, de-vig, and CSV cache (backtest baseline only)."""
from __future__ import annotations

import logging
from pathlib import Path

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


def de_vig(odds: list[float]) -> list[float]:
    """De-vig decimal odds to probabilities: p_i = (1/o_i) / Σ(1/o_j)."""
    inv = [1.0 / o for o in odds]
    total = sum(inv)
    return [v / total for v in inv]


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
        return resp.text

    def _parse_row(self, row: BeautifulSoup) -> tuple[str, float] | None:
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
