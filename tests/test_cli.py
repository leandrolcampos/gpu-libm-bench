# Copyright 2026 Leandro A. Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Tests for the benchmark command-line interface."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

SELECTIONS = ["function=expf", "metric=latency", "library=llvm-libm", "backend=cuda"]


def run_cli(tmp_path: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    executable = Path(sys.executable).with_name("gpu-libm-bench")
    # Run outside the checkout to exercise the packaged configuration search path.
    return subprocess.run(
        [executable, *arguments],
        cwd=tmp_path,
        env={**os.environ, "HYDRA_FULL_ERROR": "0"},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_configure_experiment_without_gpu(tmp_path: Path) -> None:
    result = run_cli(
        tmp_path,
        *SELECTIONS,
        "metric.sample_size=16",
        "run.seed=123",
        "run.domains=[wide,central]",
        "function.domains.central.generator.arguments.x.low=-5",
        "llvm.root=/nonexistent/llvm",
    )
    assert result.returncode == 0, result.stderr
    message, yaml = result.stdout.split("\n", 1)
    assert "execution is not available" in message
    config = OmegaConf.create(yaml)
    assert (
        config.function.name,
        config.metric.name,
        config.library.name,
        config.backend.name,
    ) == ("expf", "latency", "llvm-libm", "cuda")
    assert config.metric.sample_size == 16
    assert config.run.seed == 123
    assert config.run.domains == ["wide", "central"]
    assert config.function.domains.central.generator.name == "random"
    assert config.function.domains.central.generator.arguments.x.low == -5
    assert config.function.domains.wide.generator.arguments.x.low == -80


@pytest.mark.parametrize("selection", SELECTIONS)
def test_experiment_selections_are_required(tmp_path: Path, selection: str) -> None:
    result = run_cli(tmp_path, *(s for s in SELECTIONS if s != selection))
    assert result.returncode != 0
    assert selection.split("=")[0] in result.stderr


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("run.domains=[missing]", "run.domains"),
        ("~function.domains.wide.generator.arguments.x", "domain 'wide'"),
    ],
)
def test_inconsistent_experiment_exits_with_error(
    tmp_path: Path, override: str, message: str
) -> None:
    result = run_cli(tmp_path, *SELECTIONS, override)
    assert result.returncode != 0
    assert message in result.stderr
