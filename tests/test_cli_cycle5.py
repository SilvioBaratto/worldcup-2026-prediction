"""
Source-blind CliRunner smoke tests for Issue #22 — Cycle-5 CLI surface.

Authored from acceptance criteria only; no implementation source was read.

Design choices recorded here:
  - Command-registration checks use ``app --help`` output, which Typer
    populates for every @app.command(); this is the purest source-blind
    black-box observable.
  - Individual ``<command> --help`` tests are an additional registration
    guard: they always exit 0 for registered commands and never hit
    network or heavy compute, so they are safe even in CI.
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

Dependencies: pytest, typer[all], hypothesis
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from typer.testing import CliRunner


runner = CliRunner()


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
# Criterion 2 — ``forecast`` options and no-key default
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("option", ["--config", "--seed", "--n-simulations"])
def test_when_forecast_help_requested_expected_option_is_present(option):
    """forecast --help must advertise --config, --seed, and --n-simulations."""
    result = runner.invoke(_app(), ["forecast", "--help"])
    assert result.exit_code == 0, result.output
    assert option in result.output, (
        f"Option '{option}' not found in forecast --help:\n{result.output}"
    )


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
