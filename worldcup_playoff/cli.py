"""CLI entry point (Typer) for the FIFA World Cup 2026 forecast.

All commands live in :mod:`worldcup_playoff.cli_cycle5` and are registered onto
this app: ``forecast`` (title odds + bracket), ``backtest`` (Elo-prior tuning),
``train-hybrid``, ``build-features`` and ``fetch-live``.
"""

from __future__ import annotations

import typer

from worldcup_playoff.cli_cycle5 import register as _register_cycle5

app = typer.Typer(
    name="worldcup-playoff",
    help="FIFA World Cup 2026 forecast — Elo + Dixon-Coles Monte Carlo simulation.",
    add_completion=False,
    rich_markup_mode=None,
)

_register_cycle5(app)


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":
    main()
