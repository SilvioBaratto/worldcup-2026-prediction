"""CLI entry point using Typer.

Mirrors ``nba_playoff.cli`` adapted for FIFA World Cup 2026:

- Dataset names: teams, matches, ranking, players, match_details
  (replaces: teams, games, ranking, players, games_details)
- Round names: Round of 32 / Round of 16 / Quarter-finals / Semi-finals / Final
- ``SimulationConfig`` carries only ``{n_simulations, classifier}`` — no series fields.
- ``build_players_csv`` uses ``competition`` kwarg, not ``season``.
- Commands renamed: build-games -> build-matches, build-games-details -> build-match-details.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskID, TextColumn
from rich.table import Table

from worldcup_playoff.config import SimulationConfig, load_bracket, load_config
from worldcup_playoff.data import (
    FootballClient,
    build_match_details_csv,
    build_matches_csv,
    build_players_csv,
    build_ranking_csv,
    build_teams_csv,
    generate_bracket_toml,
)
from worldcup_playoff.pipeline import Pipeline

app = typer.Typer(
    name="worldcup-playoff",
    help="FIFA World Cup 2026 Knockout Prediction via Monte Carlo simulation.",
    add_completion=False,
    rich_markup_mode=None,
)
console = Console()


# ---------------------------------------------------------------------------
# Helpers (mirrors NBA originals exactly)
# ---------------------------------------------------------------------------


def _setup_logging(verbose: int) -> None:
    """Configure root logging level from the verbosity count flag."""
    level = {0: logging.WARNING, 1: logging.INFO}.get(verbose, logging.DEBUG)
    logging.basicConfig(
        level=level,
        format="%(levelname)-8s %(name)s: %(message)s",
    )


def _print_metrics(name: str, metrics: dict) -> None:  # type: ignore[type-arg]
    """Pretty-print classifier evaluation metrics as a Rich table.

    Args:
        name: Classifier display name used as the table title.
        metrics: Dict returned by ``ModelEvaluator.evaluate`` — must contain
            ``"classification_report"`` with per-class dicts and ``"accuracy"``.
    """
    report = metrics["classification_report"]
    table = Table(title=f"[bold]{name}[/bold]", show_lines=True)
    table.add_column("Class", style="cyan")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1-Score", justify="right")
    table.add_column("Support", justify="right")

    for label in ("0", "1"):
        if label in report:
            r = report[label]
            table.add_row(
                label,
                f"{r['precision']:.3f}",
                f"{r['recall']:.3f}",
                f"{r['f1-score']:.3f}",
                str(int(r["support"])),
            )
    table.add_row("accuracy", "", "", f"{report['accuracy']:.3f}", "", style="bold")
    console.print(table)


def _print_simulation_results(rounds: dict) -> None:  # type: ignore[type-arg]
    """Print simulation results as Rich tables, one per round.

    Args:
        rounds: Mapping of round index to ``RoundResult`` from
            ``TournamentSimulator.simulate``.
    """
    round_names = {
        0: "Round of 32",
        1: "Round of 16",
        2: "Quarter-finals",
        3: "Semi-finals",
        4: "Final",
    }

    for round_num, result in sorted(rounds.items()):
        probs = result.probabilities
        sorted_probs = sorted(
            ((team, p) for team, p in probs.items() if p > 0),
            key=lambda x: x[1],
            reverse=True,
        )

        table = Table(
            title=f"[bold]{round_names.get(round_num, f'Round {round_num}')}[/bold]"
        )
        table.add_column("Team", style="cyan")
        table.add_column("Win Probability", justify="right")

        for team, prob in sorted_probs:
            bar = "█" * int(prob * 40)
            table.add_row(team, f"{prob:6.1%} {bar}")

        console.print(table)
        console.print()


def _project_root(config: Path) -> Path:
    """Derive project root from the config file path.

    The default config lives at ``<root>/config/default.toml``, so the project
    root is two levels above the config file.  Using the *resolved* (absolute)
    path ensures the result is independent of the caller's working directory.

    Args:
        config: Path to the pipeline config TOML (e.g. ``config/default.toml``).

    Returns:
        Absolute path to the project root directory.
    """
    return config.resolve().parent.parent


_VALID_DATASETS = frozenset({"teams", "matches", "ranking", "players", "match_details"})


def _parse_season_range(seasons: str) -> tuple[int, int]:
    """Parse a ``'START-END'`` season range into ``(start_year, end_year)``.

    Args:
        seasons: String of the form ``"2006-2026"``.

    Returns:
        Tuple of ``(start_year, end_year)`` as integers.

    Raises:
        ``typer.Exit(code=1)`` on malformed input.
    """
    parts = seasons.split("-")
    if len(parts) != 2:  # noqa: PLR2004
        console.print(
            f"[red]Invalid --seasons format '{seasons}'. "
            "Use 'START-END', e.g. '2006-2026'.[/red]"
        )
        raise typer.Exit(code=1)
    try:
        start, end = int(parts[0]), int(parts[1])
    except ValueError:
        console.print(f"[red]--seasons values must be integers, got '{seasons}'.[/red]")
        raise typer.Exit(code=1)
    if start > end:
        console.print(f"[red]Start year {start} must be <= end year {end}.[/red]")
        raise typer.Exit(code=1)
    return start, end


def _parse_only_filter(only: str | None) -> set[str] | None:
    """Parse a comma-separated ``--only`` filter into a set of dataset names.

    Args:
        only: Comma-separated string such as ``"teams,matches"`` or ``None``.

    Returns:
        Set of validated dataset names, or ``None`` if *only* was not supplied.

    Raises:
        ``typer.Exit(code=1)`` if any name is not in ``_VALID_DATASETS``.
    """
    if only is None:
        return None
    names = {n.strip() for n in only.split(",") if n.strip()}
    unknown = names - _VALID_DATASETS
    if unknown:
        console.print(
            f"[red]Unknown dataset(s): {unknown}. "
            f"Valid: {sorted(_VALID_DATASETS)}[/red]"
        )
        raise typer.Exit(code=1)
    return names


def _should_run(name: str, only_set: set[str] | None, skip_details: bool) -> bool:
    """Return whether a dataset build stage should execute.

    Args:
        name: Dataset name (e.g. ``"matches"`` or ``"match_details"``).
        only_set: Restrict to this set when not ``None``.
        skip_details: When ``True``, always skip ``match_details``.

    Returns:
        ``True`` when the stage should run.
    """
    if name == "match_details" and skip_details:
        return False
    if only_set is not None and name not in only_set:
        return False
    return True


def _print_download_summary(results: dict[str, tuple[Path, int]]) -> None:
    """Print a Rich table summarising downloaded datasets.

    Args:
        results: Mapping of ``{dataset_name: (output_path, row_count)}``.
    """
    table = Table(title="[bold]Download Summary[/bold]", show_lines=True)
    table.add_column("Dataset", style="cyan")
    table.add_column("Output Path")
    table.add_column("Rows", justify="right", style="green")
    for name, (path, rows) in results.items():
        table.add_row(name, str(path), f"{rows:,}")
    console.print(table)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def download(
    seasons: str = typer.Option(
        "2006-2026",
        "--seasons",
        help='Season range as START-END (4-digit years), e.g. "2006-2026".',
    ),
    output_dir: Path = typer.Option(
        Path("dataset/csv"),
        "--output-dir",
        help="Output directory for all generated CSVs.",
    ),
    skip_details: bool = typer.Option(
        False,
        "--skip-details",
        help="Skip match_details.csv (requires matches.csv).",
    ),
    only: str | None = typer.Option(
        None,
        "--only",
        help='Comma-separated subset to build, e.g. "matches,teams".',
    ),
    verbose: int = typer.Option(0, "--verbose", "-v", count=True),
) -> None:
    """Download the full World Cup dataset from football-data.org."""
    _setup_logging(verbose)

    start_year, end_year = _parse_season_range(seasons)
    only_set = _parse_only_filter(only)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    client = FootballClient()
    results: dict[str, tuple[Path, int]] = {}
    matches_df: pd.DataFrame | None = None

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        # Stage 1: teams.csv — no season dependency
        if _should_run("teams", only_set, skip_details):
            task = progress.add_task("Building teams.csv...", total=1)
            df = build_teams_csv(output / "teams.csv", client=client)
            progress.update(task, completed=1)
            results["teams"] = (output / "teams.csv", len(df))

        # Stage 2: matches.csv — depends on season range
        if _should_run("matches", only_set, skip_details):
            task = progress.add_task(
                f"Building matches.csv ({start_year}–{end_year})...",
                total=1,
            )
            matches_df = build_matches_csv(
                output / "matches.csv",
                client=client,
                start_year=start_year,
                end_year=end_year,
            )
            progress.update(task, completed=1)
            assert matches_df is not None
            results["matches"] = (output / "matches.csv", len(matches_df))

        # Stage 3: ranking.csv — independent
        if _should_run("ranking", only_set, skip_details):
            task = progress.add_task(
                f"Building ranking.csv ({start_year}–{end_year})...",
                total=1,
            )
            df = build_ranking_csv(
                output / "ranking.csv",
                client=client,
                start_year=start_year,
                end_year=end_year,
            )
            progress.update(task, completed=1)
            results["ranking"] = (output / "ranking.csv", len(df))

        # Stage 4: players.csv — uses WC competition (not season-ranged)
        if _should_run("players", only_set, skip_details):
            task = progress.add_task("Building players.csv...", total=1)
            df = build_players_csv(
                output / "players.csv",
                client=client,
                competition="WC",
            )
            progress.update(task, completed=1)
            results["players"] = (output / "players.csv", len(df))

        # Stage 5: match_details.csv — depends on MATCH_ID from matches.csv
        if _should_run("match_details", only_set, skip_details):
            if matches_df is None:
                matches_path = output / "matches.csv"
                if not matches_path.exists():
                    console.print(
                        "[red]match_details requires matches.csv — "
                        "build it first or include 'matches' in --only.[/red]"
                    )
                    raise typer.Exit(code=1)
                matches_df = pd.read_csv(matches_path)

            assert matches_df is not None
            match_ids = [int(mid) for mid in matches_df["MATCH_ID"].unique()]
            task = progress.add_task(
                f"Building match_details.csv ({len(match_ids)} matches)...",
                total=len(match_ids),
            )
            df = build_match_details_csv(
                output / "match_details.csv",
                match_ids,
                client=client,
                checkpoint_every=100,
            )
            progress.update(task, completed=len(match_ids))
            results["match_details"] = (output / "match_details.csv", len(df))

    if results:
        _print_download_summary(results)
    else:
        console.print("[yellow]No datasets selected to download.[/yellow]")


@app.command(name="build-teams")
def build_teams(
    config: Path = typer.Option(
        "config/default.toml", "--config", "-c", help="Pipeline config"
    ),
    season: str | None = typer.Option(
        None,
        "--season",
        "-s",
        help="Ignored (football teams are queried by competition code, not season).",
    ),
    verbose: int = typer.Option(0, "--verbose", "-v", count=True),
) -> None:
    """Fetch team metadata from football-data.org and write dataset/csv/teams.csv."""
    _setup_logging(verbose)
    cfg = load_config(config)
    pipeline = Pipeline(cfg, root=_project_root(config))

    with console.status("[bold]Building teams.csv..."):
        output = pipeline.run_build_teams(season=season)

    console.print(f"[green]teams.csv saved to {output}[/green]")


@app.command(name="build-matches")
def build_matches(
    config: Path = typer.Option(
        "config/default.toml", "--config", "-c", help="Pipeline config"
    ),
    start_year: int = typer.Option(2006, "--start-year", help="First season start year"),
    end_year: int = typer.Option(2026, "--end-year", help="Last season start year"),
    verbose: int = typer.Option(0, "--verbose", "-v", count=True),
) -> None:
    """Fetch match results from football-data.org and write dataset/csv/matches.csv."""
    _setup_logging(verbose)
    cfg = load_config(config)
    pipeline = Pipeline(cfg, root=_project_root(config))

    with console.status("[bold]Building matches.csv..."):
        output = pipeline.run_build_matches(start_year=start_year, end_year=end_year)

    console.print(f"[green]matches.csv saved to {output}[/green]")


@app.command(name="build-match-details")
def build_match_details(
    config: Path = typer.Option(
        "config/default.toml", "--config", "-c", help="Pipeline config"
    ),
    matches_csv: Path = typer.Option(
        "dataset/csv/matches.csv",
        "--matches-csv",
        help="Source matches.csv for MATCH_ID column.",
    ),
    skip_details: bool = typer.Option(
        False, "--skip-details", help="Skip this command entirely."
    ),
    checkpoint_every: int = typer.Option(
        100, "--checkpoint-every", help="Checkpoint interval (matches)."
    ),
    verbose: int = typer.Option(0, "--verbose", "-v", count=True),
) -> None:
    """Fetch per-match detail stats and write dataset/csv/match_details.csv."""
    _setup_logging(verbose)

    if skip_details:
        console.print("[yellow]Skipping match_details.csv build (--skip-details)[/yellow]")
        return

    cfg = load_config(config)
    root = _project_root(config)
    matches_path = root / matches_csv

    if not matches_path.exists():
        console.print(
            f"[red]matches.csv not found at {matches_path} — build it first[/red]"
        )
        raise typer.Exit(code=1)

    matches_df = pd.read_csv(matches_path)
    match_ids = [str(int(mid)) for mid in matches_df["MATCH_ID"].unique()]

    pipeline = Pipeline(cfg, root=root)

    with console.status(
        f"[bold]Building match_details.csv — {len(match_ids)} matches...[/bold]"
    ):
        output = pipeline.run_build_match_details(
            match_ids=match_ids,
            checkpoint_every=checkpoint_every,
        )

    console.print(f"[green]match_details.csv saved to {output}[/green]")


@app.command(name="generate-bracket")
def generate_bracket(
    season: str | None = typer.Option(
        None,
        "--season",
        "-s",
        help='World Cup year, e.g. "2026". Defaults to nearest World Cup year.',
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output TOML path. Defaults to config/playoff_{year}.toml.",
    ),
    verbose: int = typer.Option(0, "--verbose", "-v", count=True),
) -> None:
    """Generate a World Cup knockout bracket TOML from football-data.org team data."""
    _setup_logging(verbose)

    from worldcup_playoff.data.bracket_builder import default_season

    resolved_season = season or default_season()
    year = resolved_season.split("-")[0] if "-" in resolved_season else resolved_season
    output_path = output or Path(f"config/playoff_{year}.toml")

    try:
        with console.status("[bold]Fetching team data..."):
            bracket = generate_bracket_toml(output_path, season=resolved_season)
        console.print(
            f"[green]Bracket saved to {output_path} "
            f"({len(bracket.matchups)} matchups)[/green]"
        )
    except (RuntimeError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)


@app.command()
def clean(
    config: Path = typer.Option(
        "config/default.toml", "--config", "-c", help="Pipeline config"
    ),
    verbose: int = typer.Option(0, "--verbose", "-v", count=True),
) -> None:
    """Clean raw match data and produce the training dataset (train_data.csv)."""
    _setup_logging(verbose)
    cfg = load_config(config)
    pipeline = Pipeline(cfg, root=_project_root(config))
    output = pipeline.run_clean()
    console.print(f"[green]Cleaned data saved to {output}[/green]")


@app.command()
def train(
    config: Path = typer.Option("config/default.toml", "--config", "-c"),
    classifier: str = typer.Option(
        "all",
        "--classifier",
        help="svm | random-forest | naive-bayes | all",
    ),
    verbose: int = typer.Option(0, "--verbose", "-v", count=True),
) -> None:
    """Train classifiers and display evaluation metrics."""
    _setup_logging(verbose)
    cfg = load_config(config)
    pipeline = Pipeline(cfg, root=_project_root(config))

    names = (
        ["svm", "random_forest", "naive_bayes"]
        if classifier == "all"
        else [classifier.replace("-", "_")]
    )

    results = pipeline.run_train(classifier_names=names)
    for name, (_, metrics) in results.items():
        _print_metrics(name, metrics)


@app.command()
def fit(
    config: Path = typer.Option("config/default.toml", "--config", "-c"),
    verbose: int = typer.Option(0, "--verbose", "-v", count=True),
) -> None:
    """Fit statistical distributions per team and save to output/distributions.json."""
    _setup_logging(verbose)
    cfg = load_config(config)
    pipeline = Pipeline(cfg, root=_project_root(config))

    with console.status("[bold]Fitting distributions for all teams..."):
        team_dists = pipeline.run_fit_distributions()

    console.print(f"[green]Distributions fitted for {len(team_dists)} teams[/green]")


@app.command()
def simulate(
    config: Path = typer.Option("config/default.toml", "--config", "-c"),
    bracket: Path = typer.Option(
        "config/playoff_2026.toml", "--bracket", "-b"
    ),
    n_simulations: int | None = typer.Option(None, "--n-simulations", "-n"),
    verbose: int = typer.Option(0, "--verbose", "-v", count=True),
) -> None:
    """Run Monte Carlo World Cup knockout simulation."""
    _setup_logging(verbose)
    cfg = load_config(config)
    bracket_cfg = load_bracket(bracket)

    if n_simulations is not None:
        cfg = cfg.model_copy(
            update={
                "simulation": SimulationConfig.model_validate(
                    {**cfg.simulation.model_dump(), "n_simulations": n_simulations}
                )
            }
        )

    pipeline = Pipeline(cfg, bracket_config=bracket_cfg, root=_project_root(config))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task: TaskID = progress.add_task(
            "Simulating knockout bracket...", total=cfg.simulation.n_simulations
        )

        def _update(n: int) -> None:
            progress.update(task, completed=n)

        rounds = pipeline.run_simulate(progress_callback=_update)

    _print_simulation_results(rounds)


@app.command()
def bracket(
    config: Path = typer.Option("config/default.toml", "--config", "-c"),
    bracket_path: Path = typer.Option(
        "config/playoff_2026.toml", "--bracket", "-b"
    ),
    n_simulations: int | None = typer.Option(None, "--n-simulations", "-n"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Output PNG path"),
    verbose: int = typer.Option(0, "--verbose", "-v", count=True),
) -> None:
    """Run simulation and render a knockout bracket visualization as a PNG."""
    _setup_logging(verbose)
    cfg = load_config(config)
    bracket_cfg = load_bracket(bracket_path)

    if n_simulations is not None:
        cfg = cfg.model_copy(
            update={
                "simulation": SimulationConfig.model_validate(
                    {**cfg.simulation.model_dump(), "n_simulations": n_simulations}
                )
            }
        )

    root = _project_root(config)
    pipeline = Pipeline(cfg, bracket_config=bracket_cfg, root=root)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task: TaskID = progress.add_task(
            "Simulating knockout bracket...", total=cfg.simulation.n_simulations
        )

        def _update(n: int) -> None:
            progress.update(task, completed=n)

        rounds = pipeline.run_simulate(progress_callback=_update)

    from worldcup_playoff.visualization.plots import ResultPlotter

    plotter = ResultPlotter(cfg.visualization)
    out = output or root / Path(cfg.visualization.output_dir) / "bracket.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plotter.plot_bracket(rounds, bracket_cfg, output_path=out)
    console.print(f"[green]Bracket saved to {out}[/green]")


@app.command()
def run(
    config: Path = typer.Option("config/default.toml", "--config", "-c"),
    bracket: Path = typer.Option(
        "config/playoff_2026.toml", "--bracket", "-b"
    ),
    verbose: int = typer.Option(0, "--verbose", "-v", count=True),
) -> None:
    """Execute the full pipeline: clean -> train -> fit -> simulate -> save probabilities.png."""
    _setup_logging(verbose)
    cfg = load_config(config)
    bracket_cfg = load_bracket(bracket)

    root = _project_root(config)
    pipeline = Pipeline(cfg, bracket_config=bracket_cfg, root=root)

    console.print("[bold]Step 1/4:[/bold] Cleaning data...")
    pipeline.run_clean()
    console.print("[green]  Done.[/green]")

    console.print("[bold]Step 2/4:[/bold] Training classifiers...")
    results = pipeline.run_train()
    for name, (_, metrics) in results.items():
        _print_metrics(name, metrics)

    console.print("[bold]Step 3/4:[/bold] Fitting distributions...")
    with console.status("[bold]Fitting distributions for all teams..."):
        team_dists = pipeline.run_fit_distributions()
    console.print(f"[green]  Fitted {len(team_dists)} teams.[/green]")

    console.print("[bold]Step 4/4:[/bold] Running simulation...")
    clf_name = cfg.simulation.classifier
    classifier, _ = results[clf_name]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task: TaskID = progress.add_task(
            "Simulating knockout bracket...", total=cfg.simulation.n_simulations
        )

        def _update(n: int) -> None:
            progress.update(task, completed=n)

        rounds = pipeline.run_simulate(
            classifier=classifier,
            team_distributions=team_dists,
            progress_callback=_update,
        )

    _print_simulation_results(rounds)

    # Save probabilities visualization
    from worldcup_playoff.visualization.plots import ResultPlotter

    plotter_dir = root / Path(cfg.visualization.output_dir)
    plotter_dir.mkdir(parents=True, exist_ok=True)
    plotter = ResultPlotter(cfg.visualization)
    plotter.plot_round_probabilities(rounds, output_path=plotter_dir / "probabilities.png")
    console.print(f"[green]Plot saved to {plotter_dir / 'probabilities.png'}[/green]")


# ---------------------------------------------------------------------------
# Cycle-5 commands (fetch-live, build-features, train-hybrid, backtest, forecast)
# Registered here after ``app`` is defined to avoid a circular-import cycle.
# ---------------------------------------------------------------------------
from worldcup_playoff.cli_cycle5 import register as _register_cycle5  # noqa: E402

_register_cycle5(app)
