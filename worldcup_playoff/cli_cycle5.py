"""Cycle-5 CLI commands: fetch-live, build-features, train-hybrid, backtest, forecast.

Registered onto the shared Typer ``app`` via ``register(app)`` called at the end
of ``cli.py`` to avoid a circular-import cycle (this module does NOT import from
``cli.py`` at module load time).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

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
    from worldcup_playoff.config import load_config, AppConfig  # noqa: PLC0415
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
        from worldcup_playoff.data.martj42_loader import load_martj42_results  # noqa: PLC0415
        from worldcup_playoff.data.elo import compute_elo  # noqa: PLC0415
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


# ---------------------------------------------------------------------------
# Command functions (module-level so Typer can inspect their signatures)
# ---------------------------------------------------------------------------


def _fetch_live(
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


def _build_features(
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


def _train_hybrid(
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


def _backtest(
    config: Path = typer.Option("config/default.toml", "--config", "-c"),
    verbose: int = typer.Option(0, "--verbose", "-v", count=True),
) -> None:
    """Run time-aware WC backtest (RPS / log-loss / Brier) vs bookmaker + legacy."""
    _setup_logging(verbose)
    import worldcup_playoff.models.evaluation as _evaluation  # noqa: PLC0415
    cfg = _safe_load_config(config)
    with _console.status("[bold]Running backtest..."):
        result = _evaluation.run_backtest(cfg=cfg, root=_root(config))
    if result is not None and not result.empty:
        _console.print(result.to_string())
    else:
        _console.print("[yellow]Backtest skipped (features+targets dataset not found).[/yellow]")


def _simulate_forecast(cfg: Any, seed: int, n_sims: int) -> Any:
    """Run Monte Carlo forecast and return the result (or None on failure)."""
    import worldcup_playoff.simulation.live_forecast as _lf  # noqa: PLC0415
    with _console.status(f"[bold]Running {n_sims:,} Monte Carlo simulations..."):
        return _lf.run_forecast(cfg, seed=seed, n_simulations=n_sims)


def _write_forecast_plots(
    result: Any, config: Path, cfg: Any, output: Path | None
) -> None:
    """Write title_odds.png and advancement.png to the resolved output directory."""
    import worldcup_playoff.visualization.forecast_plots as _fp  # noqa: PLC0415
    out_dir = Path(output) if output is not None else _root(config) / cfg.visualization.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    _fp.plot_title_odds(result, out_dir / "title_odds.png")
    _fp.plot_round_advancement(result.round_probabilities, out_dir / "advancement.png")
    _console.print(f"[green]Charts → {out_dir}[/green]")


def _forecast(
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


# ---------------------------------------------------------------------------
# Registration function — called by cli.py to avoid circular imports
# ---------------------------------------------------------------------------


def register(app: typer.Typer) -> None:
    """Register all Cycle-5 commands onto the shared Typer app."""
    app.command(name="fetch-live")(_fetch_live)
    app.command(name="build-features")(_build_features)
    app.command(name="train-hybrid")(_train_hybrid)
    app.command(name="backtest")(_backtest)
    app.command(name="forecast")(_forecast)
