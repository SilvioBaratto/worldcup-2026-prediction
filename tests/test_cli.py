"""CLI smoke tests using typer.testing.CliRunner.

All heavy computation (network, training, simulation) is mocked so tests
remain fast and fully offline.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from typer.testing import CliRunner

from worldcup_playoff.cli import _project_root, _parse_season_range, _should_run, app
from worldcup_playoff.config import AppConfig, SimulationConfig


runner = CliRunner()


# ---------------------------------------------------------------------------
# _project_root helper
# ---------------------------------------------------------------------------


class TestProjectRoot:
    def test_default_config_path_returns_project_root(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config" / "default.toml"
        config_file.parent.mkdir(parents=True)
        config_file.touch()

        result = _project_root(config_file)
        assert result == tmp_path

    def test_returns_absolute_path(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config" / "default.toml"
        config_file.parent.mkdir(parents=True)
        config_file.touch()

        result = _project_root(config_file)
        assert result.is_absolute()

    def test_resolved_against_cwd_for_relative_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "default.toml").touch()
        monkeypatch.chdir(tmp_path)

        result = _project_root(Path("config/default.toml"))
        assert result == tmp_path


# ---------------------------------------------------------------------------
# _parse_season_range helper
# ---------------------------------------------------------------------------


class TestParseSeasonRange:
    def test_valid_range(self) -> None:
        start, end = _parse_season_range("2006-2026")
        assert start == 2006
        assert end == 2026

    def test_same_year(self) -> None:
        start, end = _parse_season_range("2022-2022")
        assert start == end == 2022


# ---------------------------------------------------------------------------
# _should_run helper
# ---------------------------------------------------------------------------


class TestShouldRun:
    def test_runs_when_no_filter(self) -> None:
        assert _should_run("matches", only_set=None, skip_details=False)

    def test_skips_match_details_when_skip_details(self) -> None:
        assert not _should_run("match_details", only_set=None, skip_details=True)

    def test_runs_only_specified_dataset(self) -> None:
        assert _should_run("matches", only_set={"matches"}, skip_details=False)
        assert not _should_run("teams", only_set={"matches"}, skip_details=False)


# ---------------------------------------------------------------------------
# CLI --help smoke test
# ---------------------------------------------------------------------------


def test_app_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "worldcup-playoff" in result.output.lower() or "World Cup" in result.output


def test_clean_help() -> None:
    result = runner.invoke(app, ["clean", "--help"])
    assert result.exit_code == 0


def test_train_help() -> None:
    result = runner.invoke(app, ["train", "--help"])
    assert result.exit_code == 0


def test_simulate_help() -> None:
    result = runner.invoke(app, ["simulate", "--help"])
    assert result.exit_code == 0


def test_fit_help() -> None:
    result = runner.invoke(app, ["fit", "--help"])
    assert result.exit_code == 0


def test_bracket_help() -> None:
    result = runner.invoke(app, ["bracket", "--help"])
    assert result.exit_code == 0


def test_download_help() -> None:
    result = runner.invoke(app, ["download", "--help"])
    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# simulate command — mocked pipeline internals
# ---------------------------------------------------------------------------


def _make_minimal_rounds() -> dict:
    from worldcup_playoff.simulation.tournament import RoundResult

    return {
        0: RoundResult(counts={"Brazil": 50, "France": 50}, n_simulations=100),
    }


def _make_bracket_toml(tmp_path: Path) -> Path:
    bracket = tmp_path / "bracket.toml"
    bracket.write_text(
        'name = "Test"\n\n[[matchups]]\nhome = "Brazil"\naway = "France"\ngroup = ""\n'
    )
    return bracket


def _make_config_toml(tmp_path: Path) -> Path:
    config = tmp_path / "config.toml"
    config.write_text("[simulation]\nn_simulations = 10\n")
    return config


class TestSimulateCommand:
    def test_simulate_with_mocked_pipeline(self, tmp_path: Path) -> None:
        cfg_path = _make_config_toml(tmp_path)
        bracket_path = _make_bracket_toml(tmp_path)

        rounds = _make_minimal_rounds()

        with patch("worldcup_playoff.cli.load_config") as mock_load_cfg, \
             patch("worldcup_playoff.cli.load_bracket") as mock_load_bracket, \
             patch("worldcup_playoff.cli.Pipeline") as MockPipeline:

            mock_load_cfg.return_value = AppConfig()
            from worldcup_playoff.config import BracketConfig, Matchup
            mock_load_bracket.return_value = BracketConfig(
                matchups=[Matchup(home="Brazil", away="France")]
            )
            instance = MockPipeline.return_value
            instance.run_simulate.return_value = rounds

            result = runner.invoke(app, [
                "simulate",
                "--config", str(cfg_path),
                "--bracket", str(bracket_path),
                "--n-simulations", "10",
            ])

        assert result.exit_code == 0


class TestTrainCommand:
    def test_train_with_mocked_pipeline(self, tmp_path: Path) -> None:
        cfg_path = _make_config_toml(tmp_path)

        fake_metrics = {
            "classification_report": {
                "accuracy": 0.65,
                "0": {"precision": 0.6, "recall": 0.6, "f1-score": 0.6, "support": 20},
                "1": {"precision": 0.7, "recall": 0.7, "f1-score": 0.7, "support": 20},
            }
        }
        fake_clf = MagicMock()

        with patch("worldcup_playoff.cli.load_config") as mock_load_cfg, \
             patch("worldcup_playoff.cli.Pipeline") as MockPipeline:

            mock_load_cfg.return_value = AppConfig()
            instance = MockPipeline.return_value
            instance.run_train.return_value = {
                "naive_bayes": (fake_clf, fake_metrics)
            }

            result = runner.invoke(app, [
                "train",
                "--config", str(cfg_path),
                "--classifier", "naive-bayes",
            ])

        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# model_copy propagates validators (mirrors NBA test_cli.py issue #16)
# ---------------------------------------------------------------------------


class TestSimulateConfigMutation:
    def _apply_n_simulations(self, cfg: AppConfig, n_simulations: int) -> AppConfig:
        from pydantic import ValidationError

        return cfg.model_copy(
            update={
                "simulation": SimulationConfig.model_validate(
                    {**cfg.simulation.model_dump(), "n_simulations": n_simulations}
                )
            }
        )

    def test_valid_n_simulations_override_applied(self) -> None:
        cfg = AppConfig()
        updated = self._apply_n_simulations(cfg, 500)
        assert updated.simulation.n_simulations == 500

    def test_zero_n_simulations_raises_validation_error(self) -> None:
        from pydantic import ValidationError

        cfg = AppConfig()
        with pytest.raises(ValidationError):
            self._apply_n_simulations(cfg, 0)

    def test_negative_n_simulations_raises_validation_error(self) -> None:
        from pydantic import ValidationError

        cfg = AppConfig()
        with pytest.raises(ValidationError):
            self._apply_n_simulations(cfg, -1)

    def test_original_config_not_mutated(self) -> None:
        cfg = AppConfig()
        original = cfg.simulation.n_simulations
        self._apply_n_simulations(cfg, 99)
        assert cfg.simulation.n_simulations == original

    def test_other_simulation_fields_preserved(self) -> None:
        cfg = AppConfig()
        updated = self._apply_n_simulations(cfg, 2000)
        assert updated.simulation.classifier == cfg.simulation.classifier
