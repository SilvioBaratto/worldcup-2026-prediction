"""CLI entry point (Typer) for the FIFA World Cup 2026 forecast.

Exposes the ``worldcup-playoff`` console script (``worldcup_playoff.cli:app``)
with the commands: ``forecast`` (title odds + bracket), ``backtest`` (Elo-prior
tuning / RF-hybrid backtest), ``train-hybrid``, ``build-features`` and
``fetch-live``. Heavy modules are imported lazily inside each command so that
``--help`` stays fast and importing this module is cheap.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="worldcup-playoff",
    help="FIFA World Cup 2026 forecast — Elo + Dixon-Coles Monte Carlo simulation.",
    add_completion=False,
    rich_markup_mode=None,
)
_console = Console()


# ---------------------------------------------------------------------------
# Shared private helpers
# ---------------------------------------------------------------------------


def _setup_logging(verbose: int) -> None:
    level = {0: logging.WARNING, 1: logging.INFO}.get(verbose, logging.DEBUG)
    logging.basicConfig(level=level, format="%(levelname)-8s %(name)s: %(message)s")


def _root(config: Path) -> Path:
    return config.resolve().parent.parent


def _safe_load_config(path: Path) -> Any:
    from worldcup_playoff.config import AppConfig, load_config  # noqa: PLC0415
    try:
        return load_config(path)
    except (FileNotFoundError, OSError):
        return AppConfig()


def _load_feature_inputs(cfg: Any) -> tuple[Any, Any, Any]:
    """Return (df, elo_df, abilities) from the martj42 cache; empty values on failure."""
    import pandas as pd
    from worldcup_playoff.simulation.poisson import TeamAbilities  # noqa: PLC0415
    _empty = (
        pd.DataFrame(),
        pd.DataFrame(columns=["home_elo", "away_elo"]),
        TeamAbilities({}, {}, 0.0, 0.0, 0.0),
    )
    try:
        from worldcup_playoff.data.elo import compute_elo  # noqa: PLC0415
        from worldcup_playoff.data.martj42_loader import load_martj42_results  # noqa: PLC0415
        from worldcup_playoff.simulation.poisson import fit_dixon_coles  # noqa: PLC0415
        df = load_martj42_results(cfg.martj42)
        elo = compute_elo(df, cfg.elo)
        elo_df = pd.DataFrame(
            [{"home_elo": d.home_elo, "away_elo": d.away_elo} for d in elo.match_diffs]
        )
        return df, elo_df, fit_dixon_coles(df, cfg.poisson)
    except Exception:
        return _empty


def _print_forecast(result: Any) -> None:
    """Print title odds as a Rich table (top 20 teams by probability)."""
    table = Table(title="[bold]WC2026 Title Odds[/bold]")
    table.add_column("Team", style="cyan")
    table.add_column("Title Prob.", justify="right")
    sorted_odds = sorted(result.champion_probabilities.items(), key=lambda x: x[1], reverse=True)
    for team, prob in sorted_odds[:20]:
        if prob > 0:
            table.add_row(team, f"{prob:.2%}")
    _console.print(table)


def _simulate_forecast(cfg: Any, seed: int, n_sims: int) -> Any:
    """Run Monte Carlo forecast and return the result (or None on failure)."""
    import worldcup_playoff.simulation.live_forecast as _lf  # noqa: PLC0415
    with _console.status(f"[bold]Running {n_sims:,} Monte Carlo simulations..."):
        return _lf.run_forecast(cfg, seed=seed, n_simulations=n_sims)


def _write_forecast_plots(result: Any, config: Path, cfg: Any, output: Path | None) -> None:
    """Write bracket.png, title_odds.png and advancement.png to the output directory."""
    import worldcup_playoff.visualization.forecast_plots as _fp  # noqa: PLC0415
    out_dir = Path(output) if output is not None else _root(config) / cfg.visualization.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    _fp.plot_forecast_bracket(result, out_dir / "bracket.png", cfg.visualization)
    _fp.plot_title_odds(result, out_dir / "title_odds.png")
    _fp.plot_round_advancement(result.round_probabilities, out_dir / "advancement.png")
    _console.print(f"[green]Charts → {out_dir}[/green]")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command(name="forecast")
def forecast(
    config: Path = typer.Option("config/default.toml", "--config", "-c"),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducibility."),
    n_simulations: int | None = typer.Option(
        None, "--n-simulations", "-n",
        help="Override the number of Monte Carlo iterations.",
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o",
        help="Output directory for PNG charts; defaults to cfg.visualization.output_dir.",
    ),
    no_plots: bool = typer.Option(False, "--no-plots", help="Skip writing PNG charts."),
    verbose: int = typer.Option(0, "--verbose", "-v", count=True),
) -> None:
    """Run live WC2026 title-odds forecast (no API key required; uses martj42 schedule)."""
    _setup_logging(verbose)
    cfg = _safe_load_config(config)
    n_sims = n_simulations if n_simulations is not None else cfg.simulation.n_simulations
    result = _simulate_forecast(cfg, seed, n_sims)
    if result is None:
        return _console.print("[yellow]Forecast unavailable (data or API unreachable).[/yellow]")
    _print_forecast(result)
    if not no_plots:
        _write_forecast_plots(result, config, cfg, output)


@app.command(name="backtest")
def backtest(
    config: Path = typer.Option("config/default.toml", "--config", "-c"),
    tune_prior: bool = typer.Option(
        False, "--tune-prior",
        help="Sweep poisson.elo_prior_weight over past WCs (RPS/log-loss/Brier) "
        "and report the best weight, instead of the RF-hybrid backtest.",
    ),
    verbose: int = typer.Option(0, "--verbose", "-v", count=True),
) -> None:
    """Run time-aware WC backtest (RPS / log-loss / Brier) vs the bookmaker baseline."""
    _setup_logging(verbose)
    import worldcup_playoff.models.evaluation as _evaluation  # noqa: PLC0415
    cfg = _safe_load_config(config)
    if tune_prior:
        with _console.status("[bold]Tuning elo_prior_weight over WC2014/18/22..."):
            table = _evaluation.run_prior_tuning(cfg=cfg, root=_root(config))
        if table is not None and not table.empty:
            _console.print("[bold]Elo prior — WC2014/18/22:[/bold]")
            _console.print(table.to_string())
            best = table["rps"].idxmin()
            _console.print(
                f"[green]Best elo_prior_weight = {best} (RPS {table['rps'].min():.5f})[/green]"
            )
        else:
            _console.print("[yellow]Prior tuning skipped (martj42 results unavailable).[/yellow]")
        with _console.status("[bold]Tuning market_value_prior_weight on the 2026 groups..."):
            mv = _evaluation.run_market_value_tuning(cfg=cfg, root=_root(config))
        if mv is not None and not mv.empty:
            _console.print(
                "\n[bold]Squad-market-value prior — validated on the WC2026 group stage:[/bold]"
            )
            _console.print(mv.to_string())
            best_mv = mv["rps"].idxmin()
            _console.print(
                f"[green]Best market_value_prior_weight = {best_mv} "
                f"(RPS {mv['rps'].min():.5f}) on the 2026 group stage[/green]"
            )
        else:
            _console.print("[yellow]Market-value tuning skipped (data unavailable).[/yellow]")
        with _console.status("[bold]Joint Elo x market-value tuning over WC2018/2022..."):
            grid = _evaluation.run_2d_prior_tuning(cfg=cfg, root=_root(config))
        if grid is not None and not grid.empty:
            _console.print(
                "\n[bold]Joint Elo x market-value prior — pooled over WC2018/2022:[/bold]"
            )
            _console.print(grid.sort_values("rps").head(8).to_string(index=False))
            best = grid.loc[grid["rps"].idxmin()]
            _console.print(
                f"[green]Best (elo_prior_weight={best['elo_weight']}, "
                f"market_value_prior_weight={best['market_value_weight']}) "
                f"RPS {best['rps']:.5f}[/green]"
            )
        return
    with _console.status("[bold]Running backtest..."):
        result = _evaluation.run_backtest(cfg=cfg, root=_root(config))
    if result is not None and not result.empty:
        _console.print(result.to_string())
    else:
        _console.print("[yellow]Backtest skipped (features+targets dataset not found).[/yellow]")


@app.command(name="train-hybrid")
def train_hybrid(
    config: Path = typer.Option("config/default.toml", "--config", "-c"),
    verbose: int = typer.Option(0, "--verbose", "-v", count=True),
) -> None:
    """Fit the RF+GBM hybrid goal model from the assembled feature dataset."""
    _setup_logging(verbose)
    import worldcup_playoff.models.hybrid as _hybrid  # noqa: PLC0415
    cfg = _safe_load_config(config)
    with _console.status("[bold]Training hybrid model..."):
        model = _hybrid.train_hybrid(cfg=cfg, root=_root(config))
    if model is not None:
        _console.print("[green]Hybrid model trained successfully.[/green]")
    else:
        _console.print("[yellow]Training skipped (features dataset not found).[/yellow]")


@app.command(name="build-features")
def build_features(
    config: Path = typer.Option("config/default.toml", "--config", "-c"),
    output: Path = typer.Option("dataset/features.csv", "--output", "-o"),
    verbose: int = typer.Option(0, "--verbose", "-v", count=True),
) -> None:
    """Assemble per-match football covariates from martj42 data, Elo, and Dixon-Coles."""
    _setup_logging(verbose)
    import worldcup_playoff.features.build as _build  # noqa: PLC0415
    cfg = _safe_load_config(config)
    df, elo_df, abilities = _load_feature_inputs(cfg)
    with _console.status("[bold]Building features..."):
        result = _build.build_features(df, elo_df, abilities, config=cfg.features_build)
    if result is not None and not result.empty:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output, index=False)
        _console.print(f"[green]Features → {output} ({len(result)} rows)[/green]")
    else:
        _console.print("[yellow]No features generated (data may be missing).[/yellow]")


@app.command(name="fetch-live")
def fetch_live(
    config: Path = typer.Option("config/default.toml", "--config", "-c"),
    verbose: int = typer.Option(0, "--verbose", "-v", count=True),
) -> None:
    """Fetch live WC2026 tournament state; falls back to martj42 without an API key."""
    _setup_logging(verbose)
    import worldcup_playoff.data.live as _live  # noqa: PLC0415
    with _console.status("[bold]Fetching live tournament state..."):
        state = _live.fetch_live_data()
    if state is not None:
        _console.print(
            f"[green]Fetched {len(state.played)} played, "
            f"{len(state.remaining_group_fixtures)} remaining.[/green]"
        )
    else:
        _console.print("[green]Live data fetch complete.[/green]")


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":
    main()
