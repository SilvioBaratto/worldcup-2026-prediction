#!/usr/bin/env python3
"""Generate a 1080x1920 (9:16) brutalist thumbnail of the day's WC2026 predictions.

Follows the ../content-generator pattern: build an HTML page whose ``body`` is
sized to the canvas, then render it to PNG with headless Playwright/Chromium
(see ../content-generator/render_to_png.py).

The match list and predictions are data-driven: today's Round-of-32 fixtures are
read from the martj42 cache and each is scored with the same Elo-blended
Dixon-Coles model used by `worldcup-playoff forecast` (modal scoreline + the
Monte-Carlo advance probability).

Usage:
    python generate_thumbnail.py                  # next upcoming match day
    python generate_thumbnail.py --date 2026-06-29
    python generate_thumbnail.py -n 40000         # advance-prob simulations
    python generate_thumbnail.py -o thumbnails/today.png
"""
from __future__ import annotations

import argparse
import asyncio
import html as _html
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd
from playwright.async_api import async_playwright

from worldcup_playoff.config import AppConfig, load_config
from worldcup_playoff.data.elo import compute_elo
from worldcup_playoff.data.martj42_loader import load_martj42_results
from worldcup_playoff.simulation.knockout import _make_sampler, resolve_tie
from worldcup_playoff.data.squad_value import WC2026_SQUAD_VALUE_EUR_M
from worldcup_playoff.simulation.poisson import (
    TeamAbilities,
    blend_abilities_with_elo,
    blend_abilities_with_market_value,
    decisive_scoreline,
    fit_dixon_coles,
)

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "thumbnails"
DEFAULT_CONFIG = ROOT / "config" / "default.toml"
_WC = "FIFA World Cup"

WIDTH, HEIGHT = 1080, 1920


class MatchPrediction(TypedDict):
    """One tie's predicted modal scoreline + Monte-Carlo advance probability."""

    home: str
    away: str
    hg: int
    ag: int
    winner: str
    win_pct: int

# Flag emojis keyed by martj42 team spelling (Apple Color Emoji renders these).
_FLAGS: dict[str, str] = {
    "South Africa": "🇿🇦", "Canada": "🇨🇦", "Mexico": "🇲🇽", "South Korea": "🇰🇷",
    "Switzerland": "🇨🇭", "Bosnia and Herzegovina": "🇧🇦", "Brazil": "🇧🇷",
    "Morocco": "🇲🇦", "United States": "🇺🇸", "Australia": "🇦🇺", "Paraguay": "🇵🇾",
    "Germany": "🇩🇪", "Ivory Coast": "🇨🇮", "Ecuador": "🇪🇨", "Netherlands": "🇳🇱",
    "Japan": "🇯🇵", "Sweden": "🇸🇪", "Belgium": "🇧🇪", "Egypt": "🇪🇬",
    "Spain": "🇪🇸", "Cabo Verde": "🇨🇻", "France": "🇫🇷", "Norway": "🇳🇴",
    "Senegal": "🇸🇳", "Argentina": "🇦🇷", "Austria": "🇦🇹", "Algeria": "🇩🇿",
    "Colombia": "🇨🇴", "Portugal": "🇵🇹", "DR Congo": "🇨🇩", "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    "Croatia": "🇭🇷", "Ghana": "🇬🇭",
}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def _build_abilities(cfg: AppConfig) -> tuple[pd.DataFrame, TeamAbilities]:
    """Fit Dixon-Coles abilities, blended with the Elo and squad-market-value priors.

    Mirrors the live forecast (``live_forecast._fetch_state_and_abilities``) so the
    thumbnail uses the same model: Elo prior first, then the squad-market-value
    prior, both weighted from the config.
    """
    df = load_martj42_results(cfg.martj42)
    abilities = fit_dixon_coles(df, cfg.poisson)
    weight = getattr(cfg.poisson, "elo_prior_weight", 0.0)
    if weight > 0.0:
        elo = compute_elo(df, cfg.elo)
        abilities = blend_abilities_with_elo(abilities, elo.final_ratings, weight)
    mv_weight = getattr(cfg.poisson, "market_value_prior_weight", 0.0)
    if mv_weight > 0.0:
        abilities = blend_abilities_with_market_value(
            abilities, WC2026_SQUAD_VALUE_EUR_M, mv_weight
        )
    return df, abilities


def _predict_match(
    abilities: TeamAbilities, cfg: AppConfig, home: str, away: str, n_sims: int
) -> MatchPrediction:
    """Return the predicted decisive scoreline + Monte-Carlo advance probability."""
    rng = np.random.default_rng(cfg.simulation.random_seed)
    sampler = _make_sampler(abilities, cfg.poisson, rng)
    wins = {home: 0, away: 0}
    for _ in range(n_sims):
        w = resolve_tie(
            home, away, sampler=sampler,
            extra_time_factor=cfg.simulation.extra_time_factor,
            seed=int(rng.integers(2**32)), abilities=abilities,
            poisson_config=cfg.poisson, rng=rng,
        )
        wins[w] += 1
    winner = home if wins[home] >= wins[away] else away
    win_pct = round(max(wins.values()) / n_sims * 100)
    # Knockout ties have a winner — show the most-likely *decisive* score in the
    # advancing team's favour (no drawn predicted score).
    hg, ag = decisive_scoreline(
        abilities, home, away, home_wins=(winner == home), max_goals=cfg.poisson.max_goals
    )
    return {
        "home": home, "away": away, "hg": hg, "ag": ag,
        "winner": winner, "win_pct": win_pct,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _upcoming(df: pd.DataFrame) -> pd.DataFrame:
    """Unplayed WC2026 knockout fixtures (June 28 onward), sorted by date."""
    dates = pd.to_datetime(df["DATE"], errors="coerce")
    mask = (
        (df["TOURNAMENT"] == _WC)
        & (dates >= pd.Timestamp("2026-06-28"))
        & (df["HOME_GOALS"].isna())
    )
    out = df.loc[mask, ["HOME_TEAM", "AWAY_TEAM"]].copy()
    out["DATE"] = dates[mask].dt.normalize()
    return out.sort_values("DATE")


def _matches_for(df: pd.DataFrame, day: pd.Timestamp | None) -> tuple[pd.Timestamp, pd.DataFrame]:
    """Return (target_day, fixtures) — the given day, or the next upcoming one."""
    up = _upcoming(df)
    if up.empty:
        raise SystemExit("No upcoming WC2026 fixtures found in the martj42 cache.")
    target = day.normalize() if day is not None else up["DATE"].iloc[0]
    fixtures = up[up["DATE"] == target]
    if fixtures.empty:
        raise SystemExit(f"No WC2026 fixtures on {target.date()}.")
    return target, fixtures


# ---------------------------------------------------------------------------
# HTML (brutalist, 1080x1920) — same render-to-PNG pattern as content-generator
# ---------------------------------------------------------------------------


def _flag(team: str) -> str:
    return _FLAGS.get(team, "⚽")


def _card_html(m: MatchPrediction, hero: bool) -> str:
    home, away = m["home"], m["away"]
    hg, ag, winner = m["hg"], m["ag"], m["winner"]
    home_win = winner == home
    cls = "card hero" if hero else "card"
    return f"""
    <div class="{cls}">
      <div class="teams">
        <div class="team {'win' if home_win else 'lose'}">
          <span class="flag">{_flag(home)}</span>
          <span class="tname">{_html.escape(home).upper()}</span>
        </div>
        <div class="score">
          <span class="sc {'win' if home_win else ''}">{hg}</span>
          <span class="dash">–</span>
          <span class="sc {'win' if not home_win else ''}">{ag}</span>
        </div>
        <div class="team {'win' if not home_win else 'lose'}">
          <span class="flag">{_flag(away)}</span>
          <span class="tname">{_html.escape(away).upper()}</span>
        </div>
      </div>
      <div class="verdict">
        <span class="check">✓</span>
        <span class="wname">{_html.escape(winner).upper()}</span>
        <span class="prob">{m['win_pct']}%</span>
      </div>
    </div>"""


def build_html(day: pd.Timestamp, matches: list[MatchPrediction]) -> str:
    day_label = day.strftime("%d %B %Y").upper()
    hero_idx = max(range(len(matches)), key=lambda i: matches[i]["win_pct"])
    cards = "\n".join(_card_html(m, hero=(i == hero_idx)) for i, m in enumerate(matches))
    round_label = "ROUND OF 32"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width={WIDTH}, height={HEIGHT}">
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&display=swap');
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{
  width:{WIDTH}px; height:{HEIGHT}px; overflow:hidden;
  font-family:'Space Grotesk',-apple-system,sans-serif;
  background:#0a0a0a; position:relative;
  display:flex; flex-direction:column; align-items:center;
  justify-content:space-between; padding:55px 0 40px;
}}
.noise {{
  position:absolute; width:100%; height:100%;
  background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
  opacity:0.05; pointer-events:none; z-index:100;
}}
.pitch-marks {{
  position:absolute; inset:0;
  background-image:
    linear-gradient(90deg, transparent 96%, rgba(255,255,255,0.03) 96%),
    linear-gradient(0deg, transparent 96%, rgba(255,255,255,0.03) 96%);
  background-size:60px 60px; z-index:2;
}}
.stripe {{
  position:absolute; width:200%; left:-50%;
  transform:rotate(-6deg); transform-origin:center;
}}
.stripe-1 {{ height:360px; top:50%; margin-top:-180px;
  background:linear-gradient(90deg,#006341 0%,#00A859 50%,#006341 100%);
  box-shadow:0 0 120px rgba(0,168,89,0.45); }}
.stripe-2 {{ height:90px; top:34%; background:#E4002B; box-shadow:0 8px 0 #000; }}
.stripe-3 {{ height:26px; top:66%; background:rgba(255,255,255,0.15); }}
.geo {{ position:absolute; }}
.g1 {{ width:140px; height:140px; top:4%; right:5%; background:#FFC72C; transform:rotate(12deg); box-shadow:10px 10px 0 #000; }}
.g2 {{ width:100px; height:100px; bottom:22%; left:4%; background:#00A859; transform:rotate(-8deg); box-shadow:7px 7px 0 #000; }}
.g3 {{ width:55px; height:200px; top:12%; left:3%; background:#E4002B; transform:rotate(3deg); }}
.g4 {{ width:80px; height:80px; top:75%; right:6%; background:#FFC72C; transform:rotate(20deg); box-shadow:6px 6px 0 #E4002B; }}
.ring {{ position:absolute; border-radius:50%; }}
.r1 {{ width:220px; height:220px; bottom:25%; right:2%; border:12px solid #00A859; }}
.r2 {{ width:140px; height:140px; top:15%; left:18%; border:8px solid #FFC72C; }}
.star {{ position:absolute; font-size:50px; }}
.s1 {{ top:13%; left:43%; transform:rotate(15deg); color:#FFC72C; }}
.s2 {{ bottom:20%; right:22%; transform:rotate(-10deg); color:#00A859; }}
.s3 {{ top:71%; left:7%; transform:rotate(20deg); color:#E4002B; }}
.sticker {{ position:absolute; display:flex; align-items:center; justify-content:center;
  border-radius:50%; box-shadow:7px 7px 0 #000; }}
.k1 {{ font-size:110px; width:160px; height:160px; background:#fff; top:3%; left:6%; transform:rotate(-12deg); }}
.k2 {{ font-size:72px; width:115px; height:115px; background:#00A859; top:8%; right:26%; transform:rotate(8deg); }}
.k3 {{ font-size:96px; width:150px; height:150px; background:#fff; bottom:4%; right:6%; transform:rotate(10deg); }}
.dots {{ position:absolute; bottom:0; left:0; width:100%; height:240px;
  background-image:radial-gradient(circle, rgba(255,255,255,0.06) 2px, transparent 2px);
  background-size:30px 30px; z-index:1; }}

.head {{ text-align:center; z-index:60; width:94%; flex-shrink:0; }}
.kicker {{ display:inline-block; background:#E4002B; color:#fff; font-weight:700;
  font-size:34px; letter-spacing:8px; padding:14px 34px; transform:rotate(-2deg);
  box-shadow:8px 8px 0 #000; border:4px solid #fff; text-transform:uppercase; }}
.title {{ display:block; color:#fff; font-size:118px; font-weight:700; line-height:0.9;
  letter-spacing:-4px; margin-top:26px; text-transform:uppercase; }}
.title .y {{ color:#FFC72C; }}
.round {{ display:inline-block; margin-top:22px; color:#0a0a0a; background:#00A859;
  font-weight:700; font-size:38px; letter-spacing:6px; padding:12px 40px;
  transform:rotate(1deg); box-shadow:8px 8px 0 #000; text-transform:uppercase; }}

.content {{ z-index:55; width:90%; flex:1; display:flex; flex-direction:column;
  justify-content:center; gap:26px; padding:24px 0; }}
.card {{ background:#000; border:5px solid #fff; box-shadow:16px 16px 0 #00A859;
  padding:34px 38px; transform:rotate(-1deg); }}
.card.hero {{ box-shadow:22px 22px 0 #E4002B; transform:rotate(-1.5deg) scale(1.02); }}
.teams {{ display:flex; align-items:center; justify-content:space-between; gap:20px; }}
.team {{ flex:1; display:flex; flex-direction:column; align-items:center; gap:14px; }}
.flag {{ font-size:82px; line-height:1; }}
.tname {{ color:#fff; font-size:32px; font-weight:700; letter-spacing:1px; text-align:center; }}
.team.lose {{ opacity:0.45; }}
.team.win .tname {{ color:#FFC72C; }}
.score {{ display:flex; align-items:center; gap:18px; }}
.sc {{ font-size:104px; font-weight:700; color:#fff; line-height:0.8; }}
.sc.win {{ color:#FFC72C; }}
.dash {{ font-size:58px; color:#E4002B; font-weight:700; }}
.verdict {{ display:flex; align-items:center; justify-content:center; gap:20px;
  margin-top:28px; flex-wrap:wrap; }}
.check {{ color:#00A859; font-size:48px; font-weight:700; }}
.wname {{ color:#fff; font-size:44px; font-weight:700; letter-spacing:2px; }}
.prob {{ background:#FFC72C; color:#0a0a0a; font-size:40px; font-weight:700;
  padding:8px 24px; box-shadow:6px 6px 0 #000; }}
.aet {{ background:#E4002B; color:#fff; font-size:28px; font-weight:700; letter-spacing:3px;
  padding:8px 20px; box-shadow:5px 5px 0 #000; }}

.footer {{ flex-shrink:0; transform:rotate(2deg);
  z-index:60; background:#fff; color:#0a0a0a; font-weight:700; font-size:34px;
  letter-spacing:5px; padding:18px 46px; box-shadow:10px 10px 0 #E4002B; text-transform:uppercase; }}
.footer .ai {{ color:#E4002B; }}
.date-badge {{ position:absolute; top:0; right:0; }}
</style>
</head>
<body>
  <div class="noise"></div>
  <div class="pitch-marks"></div>
  <div class="stripe stripe-1"></div>
  <div class="stripe stripe-2"></div>
  <div class="stripe stripe-3"></div>
  <div class="geo g1"></div><div class="geo g2"></div><div class="geo g3"></div><div class="geo g4"></div>
  <div class="ring r1"></div><div class="ring r2"></div>
  <div class="dots"></div>
  <div class="star s1">✦</div><div class="star s2">✦</div><div class="star s3">✦</div>
  <div class="sticker k1">⚽</div><div class="sticker k2">📊</div><div class="sticker k3">🏆</div>

  <div class="head">
    <span class="kicker">{day_label}</span>
    <span class="title">WORLD<br>CUP <span class="y">2026</span></span>
    <span class="round">{round_label}</span>
  </div>

  <div class="content">
    {cards}
  </div>

  <div class="footer">AI <span class="ai">PREDICTION</span></div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Render (headless Chromium) — mirrors content-generator/render_to_png.py
# ---------------------------------------------------------------------------


async def _render(html: str, out_path: Path) -> None:
    html_path = out_path.with_suffix(".html")
    html_path.write_text(html, encoding="utf-8")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": WIDTH, "height": HEIGHT})
        await page.goto(f"file://{html_path.resolve()}")
        await page.wait_for_timeout(1200)  # let webfont load
        await page.screenshot(path=str(out_path), full_page=False, type="png")
        await browser.close()
    html_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", help="Target day YYYY-MM-DD (default: next upcoming match day).")
    parser.add_argument("-n", "--n-simulations", type=int, default=20000,
                        help="Advance-probability simulations per match (default 20000).")
    parser.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("-o", "--output", type=Path, help="Output PNG path.")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config.exists() else AppConfig()
    df, abilities = _build_abilities(cfg)

    day = pd.Timestamp(args.date) if args.date else None
    target, fixtures = _matches_for(df, day)

    print(f"Predicting {len(fixtures)} match(es) for {target.date()} ...")
    matches = [
        _predict_match(abilities, cfg, row.HOME_TEAM, row.AWAY_TEAM, args.n_simulations)
        for row in fixtures.itertuples(index=False)
    ]
    for m in matches:
        print(f"  {m['home']} {m['hg']}-{m['ag']} {m['away']}  -> {m['winner']} ({m['win_pct']}%)")

    OUT_DIR.mkdir(exist_ok=True)
    out_path = args.output or (OUT_DIR / f"{target.date()}.png")
    asyncio.run(_render(build_html(target, matches), out_path))
    print(f"✓ Thumbnail -> {out_path}")


if __name__ == "__main__":
    main()
