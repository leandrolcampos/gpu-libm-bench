# ===----------------------------------------------------------------------=== #
# Copyright 2026 Leandro Lacerda Campos
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Integration tests for CUDA experiments through the command-line interface."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.cuda, pytest.mark.execution]


def test_cli_prints_experiment_results_and_optional_details(tmp_path: Path) -> None:
    executable = Path(sys.executable).with_name("gpu-libm-bench")
    arguments = [
        executable,
        "function=expf",
        "metric=latency",
        "backend=cuda",
        "run.libraries=[llvm-libm]",
        "metric.sample.size=2",
        "metric.chain_length=32",
        "metric.warmup_runs=1",
    ]

    for show_details in (False, True):
        result = subprocess.run(
            [*arguments, *(["--details"] if show_details else [])],
            cwd=tmp_path,
            env={**os.environ, "HYDRA_FULL_ERROR": "0", "COLUMNS": "140"},
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

        assert result.returncode == 0, result.stderr

        assert "float expf(float x)" in result.stdout
        assert "Latency (ns/call)" in result.stdout
        assert "cuda — 0:" in result.stdout
        assert "Domain: central" in result.stdout
        assert "Domain: wide" in result.stdout
        assert "llvm-libm" in result.stdout
        assert ("Configuration and Provenance" in result.stdout) == show_details
