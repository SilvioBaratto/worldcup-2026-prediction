"""Structural tests for the squad-market-value prior tuning."""

from __future__ import annotations

import numpy as np
import pandas as pd

from worldcup_playoff.config import AppConfig
from worldcup_playoff.models.evaluation import backtest_market_value_weight


def _results() -> pd.DataFrame:
    """A small coerced-martj42 frame: pre-2026 friendlies + two 2026 group ties."""
    teams = ["A", "B", "C", "D"]
    rng = np.random.default_rng(0)
    rows: list[dict[str, object]] = []
    # Double round-robin of friendlies through 2024–2025 so every team is fitted.
    for rep in range(3):
        for i, h in enumerate(teams):
            for a in teams[i + 1 :]:
                rows.append({
                    "DATE": f"2025-0{rep + 1}-1{i}",
                    "HOME_TEAM": h, "AWAY_TEAM": a,
                    "HOME_GOALS": int(rng.integers(0, 4)),
                    "AWAY_GOALS": int(rng.integers(0, 3)),
                    "TOURNAMENT": "Friendly",
                })
    # Two real 2026 WC group matches to score against.
    rows.append({"DATE": "2026-06-15", "HOME_TEAM": "A", "AWAY_TEAM": "B",
                 "HOME_GOALS": 2, "AWAY_GOALS": 0, "TOURNAMENT": "FIFA World Cup"})
    rows.append({"DATE": "2026-06-16", "HOME_TEAM": "C", "AWAY_TEAM": "D",
                 "HOME_GOALS": 1, "AWAY_GOALS": 1, "TOURNAMENT": "FIFA World Cup"})
    return pd.DataFrame(rows)


def test_market_value_tuning_returns_weight_indexed_metrics():
    tbl = backtest_market_value_weight(
        _results(), AppConfig(), {"A": 1000.0, "B": 10.0, "C": 500.0, "D": 50.0},
        weights=(0.0, 0.3, 0.5),
    )
    assert list(tbl.index) == [0.0, 0.3, 0.5]
    assert {"rps", "log_loss", "brier", "n_matches"} <= set(tbl.columns)
    assert (tbl["n_matches"] == 2).all()
    assert tbl["rps"].notna().all()


def test_market_value_tuning_empty_when_no_2026_group_matches():
    df = _results()
    df = df[df["TOURNAMENT"] == "Friendly"]  # drop the WC2026 rows
    tbl = backtest_market_value_weight(df, AppConfig(), {"A": 1.0, "B": 1.0}, weights=(0.0,))
    assert tbl.empty
