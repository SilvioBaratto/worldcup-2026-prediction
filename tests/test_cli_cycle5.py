"""
Source-blind CliRunner smoke tests for Cycle-5 CLI surface.

Originally authored for Issue #22; extended in Issue #50 with:
  - Fix for brittle ANSI option-presence tests (NO_COLOR=1 per-invocation).
  - Tests for ``--output`` and ``--no-plots`` options on ``forecast``.
  - Canonical CliRunner test: both PNGs written for a stub ForecastResult.
  - Graceful-message test for the None/no-key branch.
  - ``--no-plots`` suppresses PNG writing.
  - ``backtest`` surfaces rps_bookmaker when stub odds are provided.

Authored from acceptance criteria only; no implementation source was read.

Design choices recorded here:
  - Command-registration checks use ``app --help`` output, which Typer
    populates for every @app.command(); this is the purest source-blind
    black-box observable.
  - Individual ``<command> --help`` tests are an additional registration
    guard: they always exit 0 for registered commands and never hit
    network or heavy compute, so they are safe even in CI.
  - Option-presence assertions (``--config``, ``--seed``, ``--no-plots`` …)
    pass ``env=_NO_ANSI`` so Rich does not inject ANSI escape codes that
    break literal-substring matching on CI.  The options ARE present in the
    help text; only the rendering differs.  See Issue #50 audit §A.
  - Patch targets for full-invocation smoke tests are derived from the
    project-structure in requirements.md.  The CLI is expected to delegate
    to those modules; deviating from the spec causes these tests to fail
    (which is the correct outcome — the implementation must match the spec).
  - "no-key by default" (criterion 2): ``forecast`` is invoked with
    FOOTBALL_DATA_API_KEY removed from the environment.
  - Property-based tests (Hypothesis) cover the re-runnable invariant
    (criterion 2): ``forecast --seed <s>`` and
    ``forecast --n-simulations <n>`` must exit 0 for all valid inputs.
  - Criterion 5 (SOLID / code-quality) is classified NOT VERIFIABLE by
    the oracle and is deliberately omitted.

Dependencies: pytest, typer[all], hypothesis, matplotlib
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import matplotlib

matplotlib.use("Agg")  # headless backend — must be set before any pyplot import

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from typer.testing import CliRunner


runner = CliRunner()

# Suppress Rich/Typer ANSI escapes so literal ``--option`` strings are testable.
# See Issue #50 audit §A: without this, ``--config`` is rendered as separate
# ANSI spans and the substring check fails on CI where Rich is fully functional.
_NO_ANSI: dict[str, str] = {"NO_COLOR": "1"}


def _app():
    """Return the Typer app from the canonical entry-point (lazy import)."""
    from worldcup_playoff.cli import app  # noqa: PLC0415

    return app


# ═══════════════════════════════════════════════════════════════════════════════
# Criterion 1 — five new commands must be registered in the CLI
# ═══════════════════════════════════════════════════════════════════════════════

NEW_COMMANDS = [
    "fetch-live",
    "build-features",
    "train-hybrid",
    "backtest",
    "forecast",
]


@pytest.mark.parametrize("cmd", NEW_COMMANDS)
def test_when_help_requested_new_command_is_listed(cmd):
    """Top-level --help output must include the new command name."""
    result = runner.invoke(_app(), ["--help"])
    assert result.exit_code == 0, result.output
    assert cmd in result.output, f"Expected '{cmd}' in help output but got:\n{result.output}"


@pytest.mark.parametrize("cmd", NEW_COMMANDS)
def test_when_new_command_help_requested_it_exits_zero(cmd):
    """
    ``<command> --help`` exits 0 for every new command.

    This is the minimal no-compute smoke test: a command that is not registered
    exits 2 ("No such command"); exit 0 proves registration.
    """
    result = runner.invoke(_app(), [cmd, "--help"])
    assert result.exit_code == 0, f"'{cmd} --help' exited {result.exit_code}:\n{result.output}"


# ═══════════════════════════════════════════════════════════════════════════════
# Criterion 3 — existing commands must remain registered and unchanged
# ═══════════════════════════════════════════════════════════════════════════════

EXISTING_COMMANDS = [
    "download",
    "clean",
    "train",
    "fit",
    "simulate",
    "bracket",
    "run",
]


@pytest.mark.parametrize("cmd", EXISTING_COMMANDS)
def test_when_help_requested_existing_command_is_still_listed(cmd):
    """Pre-Cycle-5 commands must still appear in the top-level help."""
    result = runner.invoke(_app(), ["--help"])
    assert result.exit_code == 0, result.output
    assert cmd in result.output, (
        f"Pre-existing '{cmd}' missing from help; backward-compat broken.\nOutput:\n{result.output}"
    )


@pytest.mark.parametrize("cmd", EXISTING_COMMANDS)
def test_when_existing_command_help_requested_it_exits_zero(cmd):
    """Each pre-existing command must still accept --help without error."""
    result = runner.invoke(_app(), [cmd, "--help"])
    assert result.exit_code == 0, f"'{cmd} --help' exited {result.exit_code}:\n{result.output}"


# ═══════════════════════════════════════════════════════════════════════════════
# Criterion 2 — ``forecast`` options: --config / --seed / --n-simulations
#
# FIX (Issue #50 §A): pass env=_NO_ANSI so Rich does not colorize the ``--``
# dashes into separate ANSI spans.  The literal ``--config`` substring then
# matches on both local (broken Rich) and CI (working Rich) environments.
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("option", ["--config", "--seed", "--n-simulations"])
def test_when_forecast_help_requested_expected_option_is_present(option):
    """forecast --help must advertise --config, --seed, and --n-simulations."""
    result = runner.invoke(_app(), ["forecast", "--help"], env=_NO_ANSI)
    assert result.exit_code == 0, result.output
    assert option in result.output, (
        f"Option '{option}' not found in forecast --help:\n{result.output}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Criterion 2 — ``forecast`` gains ``--output`` and ``--no-plots`` options
# (Issue #50 additions)
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("option", ["--output", "--no-plots"])
def test_when_forecast_help_requested_new_option_is_present(option):
    """
    forecast --help must list the --output and --no-plots options added in
    Cycle 5 (Issue #50).  --output overrides cfg.visualization.output_dir;
    --no-plots suppresses PNG rendering.
    """
    result = runner.invoke(_app(), ["forecast", "--help"], env=_NO_ANSI)
    assert result.exit_code == 0, result.output
    assert option in result.output, (
        f"Option {option!r} not found in forecast --help:\n{result.output}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Criterion 2 — ``forecast`` no-key default and re-runnable
# ═══════════════════════════════════════════════════════════════════════════════


def test_when_forecast_invoked_without_api_key_it_exits_zero():
    """
    forecast must not require FOOTBALL_DATA_API_KEY (no-key default).

    The martj42 schedule is the declared default source per the spec, so the
    command must succeed without the football-data.org API key present in the
    environment.
    """
    clean_env = {k: v for k, v in os.environ.items() if k != "FOOTBALL_DATA_API_KEY"}
    with (
        patch(
            "worldcup_playoff.simulation.live_forecast.run_forecast",
            MagicMock(return_value=None),
        ),
        patch.dict(os.environ, clean_env, clear=True),
    ):
        result = runner.invoke(_app(), ["forecast"])
    assert result.exit_code == 0, (
        f"forecast exited {result.exit_code} without an API key:\n{result.output}"
    )


def test_when_forecast_rerun_both_invocations_exit_zero():
    """
    forecast is re-runnable: two consecutive invocations must both exit 0.

    The spec states "re-runnable"; this test pins that a second call does not
    fail on stale state, locked files, or unconditional overwrite errors.
    """
    mock_fn = MagicMock(return_value=None)
    with patch("worldcup_playoff.simulation.live_forecast.run_forecast", mock_fn):
        first = runner.invoke(_app(), ["forecast"])
        second = runner.invoke(_app(), ["forecast"])
    assert first.exit_code == 0, f"First run failed:\n{first.output}"
    assert second.exit_code == 0, f"Second run failed:\n{second.output}"


def test_when_forecast_returns_none_then_graceful_message_is_in_output():
    """
    The None / no-key branch must print a non-empty graceful message.

    Criterion: "the None/no-key branch still prints the graceful message and
    exits 0."  The message must be non-empty and must not be an exception
    traceback.
    """
    with patch("worldcup_playoff.simulation.live_forecast.run_forecast", return_value=None):
        result = runner.invoke(_app(), ["forecast"])
    assert result.exit_code == 0, f"Expected exit 0, got {result.exit_code}:\n{result.output}"
    stripped = result.output.strip()
    assert len(stripped) > 0, (
        "Expected a graceful message when forecast returns None, but stdout was empty"
    )
    assert "Traceback" not in stripped, "Expected a graceful message but got an exception traceback"


# ═══════════════════════════════════════════════════════════════════════════════
# Criterion: forecast, on a non-None result, writes title_odds.png and
#            advancement.png under --output via plot_title_odds /
#            plot_round_advancement  (Issue #50 canonical CliRunner test)
# ═══════════════════════════════════════════════════════════════════════════════


def _make_stub_forecast_result() -> SimpleNamespace:
    """
    Minimal duck-typed ForecastResult for CLI tests.

    Attributes mirror the ForecastResult contract verified in test_live_forecast.py:
    ``champion_probabilities: dict[str, float]`` and
    ``round_probabilities: dict[str, dict[str, float]]``.
    Using SimpleNamespace keeps this test source-blind (no import from src/).
    """
    return SimpleNamespace(
        champion_probabilities={
            "Brazil": 0.40,
            "France": 0.35,
            "Spain": 0.25,
        },
        round_probabilities={
            "Round of 32": {"Brazil": 0.9, "France": 0.85, "Spain": 0.8},
            "Final": {"Brazil": 0.40, "France": 0.35, "Spain": 0.25},
        },
    )


def test_when_forecast_called_with_stub_result_then_both_pngs_are_written(tmp_path: Path):
    """
    Canonical CliRunner test (Issue #50 criterion):
    a stub ForecastResult must cause both title_odds.png and advancement.png to
    be written under the --output directory.

    ``run_forecast`` is patched at the source module so the CLI receives the
    stub rather than hitting the live API.  The actual plot functions run on
    the matplotlib Agg backend (set at module level) — this tests the full
    CLI→plot pipeline without mocking visualization.

    Patch: worldcup_playoff.simulation.live_forecast.run_forecast
    (source-level; works when CLI uses module-qualified calls or lazy imports).
    """
    with patch(
        "worldcup_playoff.simulation.live_forecast.run_forecast",
        return_value=_make_stub_forecast_result(),
    ):
        result = runner.invoke(_app(), ["forecast", "--output", str(tmp_path)])

    assert result.exit_code == 0, f"forecast exited {result.exit_code}:\n{result.output}"
    assert (tmp_path / "title_odds.png").exists(), (
        f"title_odds.png was not written to {tmp_path}.\nCLI output:\n{result.output}"
    )
    assert (tmp_path / "advancement.png").exists(), (
        f"advancement.png was not written to {tmp_path}.\nCLI output:\n{result.output}"
    )


def test_when_forecast_returns_non_none_result_then_title_odds_png_is_written(tmp_path: Path):
    """
    forecast must write title_odds.png via plot_title_odds when result is not None.
    One assertion per criterion — paired with the advancement test below.
    """
    with patch(
        "worldcup_playoff.simulation.live_forecast.run_forecast",
        return_value=_make_stub_forecast_result(),
    ):
        result = runner.invoke(_app(), ["forecast", "--output", str(tmp_path)])
    assert result.exit_code == 0, f"forecast exited {result.exit_code}:\n{result.output}"
    assert (tmp_path / "title_odds.png").exists(), (
        "title_odds.png was not written even though forecast returned a non-None result"
    )


def test_when_forecast_returns_non_none_result_then_advancement_png_is_written(tmp_path: Path):
    """
    forecast must write advancement.png via plot_round_advancement when result is not None.
    """
    with patch(
        "worldcup_playoff.simulation.live_forecast.run_forecast",
        return_value=_make_stub_forecast_result(),
    ):
        result = runner.invoke(_app(), ["forecast", "--output", str(tmp_path)])
    assert result.exit_code == 0, f"forecast exited {result.exit_code}:\n{result.output}"
    assert (tmp_path / "advancement.png").exists(), (
        "advancement.png was not written even though forecast returned a non-None result"
    )


def test_when_forecast_no_plots_flag_used_then_no_pngs_are_written(tmp_path: Path):
    """
    When --no-plots is passed, neither PNG must be written even when forecast
    returns a non-None result.
    """
    with patch(
        "worldcup_playoff.simulation.live_forecast.run_forecast",
        return_value=_make_stub_forecast_result(),
    ):
        result = runner.invoke(_app(), ["forecast", "--output", str(tmp_path), "--no-plots"])
    assert result.exit_code == 0, f"forecast --no-plots exited {result.exit_code}:\n{result.output}"
    assert not (tmp_path / "title_odds.png").exists(), (
        "title_odds.png must not be written when --no-plots is specified"
    )
    assert not (tmp_path / "advancement.png").exists(), (
        "advancement.png must not be written when --no-plots is specified"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Criterion 4 — smoke tests: exit code 0 against monkeypatched module calls
#
# Patch targets are derived from requirements.md project structure:
#   fetch-live     → worldcup_playoff.data.live
#   build-features → worldcup_playoff.features.build
#   train-hybrid   → worldcup_playoff.models.hybrid
#   backtest       → worldcup_playoff.models.evaluation
#   forecast       → worldcup_playoff.simulation.live_forecast
#
# Function names follow the module's declared responsibility; if the
# implementation uses different names it should either align with the spec
# or update these targets accordingly.
# ═══════════════════════════════════════════════════════════════════════════════


def test_when_fetch_live_invoked_it_exits_zero():
    """fetch-live exits 0 when the live-data module call is monkeypatched."""
    with patch(
        "worldcup_playoff.data.live.fetch_live_data",
        MagicMock(return_value=None),
    ):
        result = runner.invoke(_app(), ["fetch-live"])
    assert result.exit_code == 0, result.output


def test_when_build_features_invoked_it_exits_zero():
    """build-features exits 0 when the feature-builder call is monkeypatched."""
    with patch(
        "worldcup_playoff.features.build.build_features",
        MagicMock(return_value=None),
    ):
        result = runner.invoke(_app(), ["build-features"])
    assert result.exit_code == 0, result.output


def test_when_train_hybrid_invoked_it_exits_zero():
    """train-hybrid exits 0 when the hybrid-training call is monkeypatched."""
    with patch(
        "worldcup_playoff.models.hybrid.train_hybrid",
        MagicMock(return_value=None),
    ):
        result = runner.invoke(_app(), ["train-hybrid"])
    assert result.exit_code == 0, result.output


def test_when_backtest_invoked_it_exits_zero():
    """backtest exits 0 when the evaluation call is monkeypatched."""
    with patch(
        "worldcup_playoff.models.evaluation.run_backtest",
        MagicMock(return_value=None),
    ):
        result = runner.invoke(_app(), ["backtest"])
    assert result.exit_code == 0, result.output


def test_when_forecast_invoked_it_exits_zero():
    """forecast exits 0 when the live-forecast call is monkeypatched."""
    with patch(
        "worldcup_playoff.simulation.live_forecast.run_forecast",
        MagicMock(return_value=None),
    ):
        result = runner.invoke(_app(), ["forecast"])
    assert result.exit_code == 0, result.output


# ═══════════════════════════════════════════════════════════════════════════════
# Criterion: backtest surfaces the bookmaker baseline when odds are available
# (Issue #50, consumes #48)
# ═══════════════════════════════════════════════════════════════════════════════


def test_when_backtest_invoked_with_stub_odds_then_bookmaker_baseline_is_surfaced():
    """
    backtest CLI must print/display the bookmaker baseline (rps_bookmaker) in
    its output when stubbed match odds are available.

    run_backtest is stubbed to return a DataFrame that already contains
    rps_bookmaker; the CLI is expected to print or render this result so the
    user can see the bookmaker baseline alongside the model metrics.

    Patch: worldcup_playoff.models.evaluation.run_backtest.
    """
    import pandas as pd

    bh_result = pd.DataFrame(
        {
            "rps_hybrid": [0.19],
            "rps_legacy": [0.23],
            "rps_bookmaker": [0.17],
        },
        index=[2018],
    )
    with patch(
        "worldcup_playoff.models.evaluation.run_backtest",
        return_value=bh_result,
    ):
        result = runner.invoke(_app(), ["backtest"])

    assert result.exit_code == 0, f"backtest exited {result.exit_code}:\n{result.output}"
    assert "rps_bookmaker" in result.output or "bookmaker" in result.output.lower(), (
        "backtest CLI must surface the bookmaker baseline in its output when odds are available.\n"
        f"Actual output:\n{result.output}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Property-based: re-runnable invariant for ``forecast`` (criterion 2)
#
# Invariant A: forecast --seed <s> exits 0 for all s in [0, 2^31-1]
# Invariant B: forecast --n-simulations <n> exits 0 for all n >= 1
# ═══════════════════════════════════════════════════════════════════════════════


@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(max_examples=10)
def test_when_forecast_given_any_valid_seed_it_exits_zero(seed):
    """
    For any non-negative integer seed, forecast must exit 0.

    Invariant: forecast(--seed s) → exit 0  ∀ s ∈ [0, 2^31-1]
    """
    with patch(
        "worldcup_playoff.simulation.live_forecast.run_forecast",
        MagicMock(return_value=None),
    ):
        result = runner.invoke(_app(), ["forecast", "--seed", str(seed)])
    assert result.exit_code == 0, (
        f"forecast --seed {seed} exited {result.exit_code}:\n{result.output}"
    )


@given(n=st.integers(min_value=1, max_value=200_000))
@settings(max_examples=10)
def test_when_forecast_given_any_positive_n_simulations_it_exits_zero(n):
    """
    For any positive n-simulations value, forecast must exit 0.

    Invariant: forecast(--n-simulations n) → exit 0  ∀ n ≥ 1
    """
    with patch(
        "worldcup_playoff.simulation.live_forecast.run_forecast",
        MagicMock(return_value=None),
    ):
        result = runner.invoke(_app(), ["forecast", "--n-simulations", str(n)])
    assert result.exit_code == 0, (
        f"forecast --n-simulations {n} exited {result.exit_code}:\n{result.output}"
    )
