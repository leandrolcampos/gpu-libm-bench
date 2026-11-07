# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Tests for the benchmark command-line interface."""

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf

from gpu_libm_bench import cli
from gpu_libm_bench.experiment import Experiment

SELECTIONS = [
    "function=expf",
    "metric=latency",
    "backend=cuda",
]


def _run_cli(tmp_path: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    executable = Path(sys.executable).with_name("gpu-libm-bench")
    # Run outside the checkout to exercise the package configuration search path.
    return subprocess.run(
        [executable, *arguments],
        cwd=tmp_path,
        env={**os.environ, "HYDRA_FULL_ERROR": "0"},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_help_describes_report_and_hydra_options(tmp_path: Path) -> None:
    result = _run_cli(tmp_path, "--help")

    assert result.returncode == 0, result.stderr
    assert "--details" in result.stdout
    assert "--cfg job --resolve" in result.stdout
    assert "--hydra-help" in result.stdout


def test_config_can_be_inspected_without_running_experiment(tmp_path: Path) -> None:
    result = _run_cli(
        tmp_path,
        "--cfg",
        "job",
        "--resolve",
        "--details",
        *SELECTIONS,
        "metric.sample_size=16",
        "llvm.root=/nonexistent/llvm",
    )
    assert result.returncode == 0, result.stderr

    config = OmegaConf.create(result.stdout)

    assert (
        config.function.name,
        config.metric.name,
        config.libraries["llvm-libm"].name,
        config.backend.name,
    ) == ("expf", "latency", "llvm-libm", "cuda")
    assert config.metric.sample_size == 16
    assert config.llvm.root == "/nonexistent/llvm"


@pytest.mark.parametrize("show_details", [False, True])
def test_cli_runs_and_reports_experiment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, show_details: bool
) -> None:
    experiment = Mock(spec=Experiment)
    factory = Mock(return_value=experiment)
    report = Mock()
    monkeypatch.setattr(cli, "Experiment", factory)
    monkeypatch.setattr(cli, "print_result", report)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gpu-libm-bench",
            *SELECTIONS,
            *(["--details"] if show_details else []),
            "metric.sample_size=16",
            "run.domains=[wide,central]",
        ],
    )

    cli.main()

    factory.assert_called_once()
    config = factory.call_args.args[0]
    assert config.function.name == "expf"
    assert config.metric.sample_size == 16
    assert config.run.domains == ["wide", "central"]

    experiment.run.assert_called_once_with()
    report.assert_called_once_with(
        experiment.run.return_value, show_details=show_details
    )


@pytest.mark.parametrize("selection", SELECTIONS)
def test_experiment_selections_are_required(tmp_path: Path, selection: str) -> None:
    result = _run_cli(tmp_path, *(s for s in SELECTIONS if s != selection))

    assert result.returncode != 0
    assert selection.split("=")[0] in result.stderr


def test_invalid_configuration_exits_with_diagnostic(tmp_path: Path) -> None:
    result = _run_cli(tmp_path, *SELECTIONS, "run.libraries=[missing]")

    assert result.returncode != 0
    assert "run.libraries contains an unknown library: 'missing'" in result.stderr


def test_execution_failure_is_reported_by_hydra(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    experiment = Mock(spec=Experiment)
    experiment.run.side_effect = RuntimeError("kernel compilation failed")
    report = Mock()
    monkeypatch.setattr(cli, "Experiment", Mock(return_value=experiment))
    monkeypatch.setattr(cli, "print_result", report)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["gpu-libm-bench", *SELECTIONS])
    monkeypatch.setenv("HYDRA_FULL_ERROR", "0")

    with pytest.raises(SystemExit) as caught:
        cli.main()

    assert caught.value.code == 1
    assert "RuntimeError: kernel compilation failed" in capsys.readouterr().err
    report.assert_not_called()
