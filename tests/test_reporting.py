# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Tests for terminal benchmark reports and aggregate-only presentation."""

from dataclasses import replace
from io import StringIO
from pathlib import Path

import numpy as np
import pytest
from rich.console import Console

from gpu_libm_bench import inputs
from gpu_libm_bench.benchmarks.latency import LatencyMeasurement, LatencyResult
from gpu_libm_bench.compiler.images import ImageBuildInfo
from gpu_libm_bench.compiler.targets import CompilationTarget
from gpu_libm_bench.configs import (
    ArgumentConfig,
    BackendConfig,
    DomainConfig,
    ExperimentConfig,
    FunctionConfig,
    LatencyConfig,
    LibraryArtifactConfig,
    LibraryConfig,
    LLVMConfig,
    RandomGeneratorConfig,
    RunConfig,
    SignatureConfig,
)
from gpu_libm_bench.devices import DeviceImage, DeviceInfo, TimerInfo
from gpu_libm_bench.experiment import Experiment, ExperimentResult, LatencyResults
from gpu_libm_bench.reporting.terminal import print_result


@pytest.fixture
def experiment() -> ExperimentResult:
    target = CompilationTarget("cuda", "nvptx64-nvidia-cuda", "sm_120")
    device_info = DeviceInfo(
        "cuda", 0, "test GPU", "sm_120", 2**30, TimerInfo("%globaltimer", 64, 10**9)
    )

    domains = {
        name: DomainConfig(
            RandomGeneratorConfig(
                "random",
                {"x": {"distribution": "uniform", "low": -bound, "high": bound}},
            )
        )
        for name, bound in (("central", 10.0), ("wide", 80.0))
    }
    config = ExperimentConfig(
        FunctionConfig(
            "expf",
            SignatureConfig("float", [ArgumentConfig("x", "float", "input")]),
            domains,
        ),
        LatencyConfig("latency", "ns/call", 2, 512, 3, 10),
        LibraryConfig("llvm-libm", {"cuda": LibraryArtifactConfig("/libm.bc")}),
        BackendConfig("cuda", target.target_triple, target.architecture, 0, "/cuda"),
        RunConfig(42, None),
        LLVMConfig("/llvm"),
    )

    image_build_info = ImageBuildInfo(
        Path("/llvm/bin/clang++"),
        "clang version 24.0.0",
        "CUDA Toolkit",
        "13.4",
        Path("/cuda"),
        Path("/libm.bc"),
        DeviceImage(Path("/math.cubin"), target, "llvm-libm", ("expf",), None),
        ("-O3", "-nogpulib"),
        ("-nostdlib",),
    )

    results = {}
    for name in domains:
        measurement = LatencyMeasurement(
            inputs.InputBatch(
                name, {"x": np.array([1234567, 7654321], dtype=np.float32)}
            ),
            np.full((2, 3), 987654321, dtype=np.uint64),
            device_info,
            2,
            512,
            10,
        )

        results[name] = LatencyResult(
            measurement,
            np.array([10.123456789, 14.123456789]),
            12.123456789,
            stdev=8**0.5,
            min=10.123456789,
            max=14.123456789,
        )

    return ExperimentResult(
        config,
        device_info,
        image_build_info,
        {"Python": "3.12", "NumPy": "2.0"},
        Experiment(config).generator_descriptions,
        LatencyResults(tuple(results.values())),
    )


def _render(
    result: ExperimentResult, *, width: int = 140, show_details: bool = False
) -> str:
    output = StringIO()
    print_result(
        result,
        show_details=show_details,
        console=Console(file=output, width=width, color_system=None),
    )
    return output.getvalue()


def test_report_presents_domain_statistics_and_provenance(
    experiment: ExperimentResult,
) -> None:
    output = _render(experiment, show_details=True)

    assert output.count("float expf(float x)") == 1
    assert output.count("Latency (ns/call)") == 1
    assert output.count("Domain: central") == output.count("Domain: wide") == 1
    assert "x: uniform(-10, 10)" in output

    central = output.split("Domain: central", 1)[1].split("Domain: wide", 1)[0]
    labels = next(line for line in central.splitlines() if "StDev" in line)
    row = next(line for line in central.splitlines() if "12.123" in line)
    assert [cell.strip() for cell in labels.split("│")[1:-1]] == [
        "Library",
        "Mean",
        "StDev",
        "Min",
        "Max",
        "Samples",
    ]
    assert [cell.strip() for cell in row.split("│")[1:-1]] == [
        "llvm-libm",
        "12.123",
        "2.8284",
        "10.123",
        "14.123",
        "2",
    ]

    comparisons, provenance = output.split("Configuration and Provenance", 1)
    assert "Method:" not in comparisons and "Compiler:" not in comparisons
    assert "CUDA — 0: test GPU (sm_120)" in comparisons
    assert "Memory:" not in comparisons and "Timer:" not in comparisons
    assert provenance.count("Memory: 1,024.00 MiB") == 1
    assert provenance.count("Timer: %globaltimer, 64 bits, 1000000000 Hz") == 1
    assert provenance.count("Compiler:") == 1
    assert provenance.count("Repetitions: 3; chain length: 512") == 1
    assert provenance.count("Seed: 42") == 1
    assert "Domains: central, wide" in provenance
    assert "Library: llvm-libm" in provenance

    for metadata in (
        "NumPy=2.0",
        "clang version 24.0.0",
        "CUDA Toolkit 13.4",
        "/libm.bc",
        "/math.cubin",
        "-nogpulib",
    ):
        assert metadata in provenance

    for raw_value in ("1234567", "7654321", "987654321", "10.123456789"):
        assert raw_value not in output


def test_report_omits_provenance_by_default(experiment: ExperimentResult) -> None:
    output = _render(experiment)

    assert "Domain: central" in output and "Domain: wide" in output
    assert "Configuration and Provenance" not in output
    assert "Compiler:" not in output and "Memory:" not in output
    assert "Timer:" not in output and "Repetitions:" not in output


def test_domain_tables_preserve_widths_and_values(experiment: ExperimentResult) -> None:
    wide = replace(
        experiment.results["wide"],
        mean=12345678.9,
        stdev=765432.1,
        min=10000000.0,
        max=15000000.0,
    )
    large = replace(
        experiment, results=LatencyResults((experiment.results["central"], wide))
    )

    output = _render(large, width=40)

    headers = [line for line in output.splitlines() if "StDev" in line]
    assert len(headers) == 2
    assert headers[0] == headers[1]

    row = next(line for line in output.splitlines() if "1.2346e+07" in line)
    assert len(row) > 40
    assert "7.6543e+05" in row and "1.5e+07" in row


def test_report_uses_recorded_generator_descriptions(
    experiment: ExperimentResult,
) -> None:
    descriptions = {
        name: f"x: recorded generator for {name}\nrange: recorded range for {name}"
        for name in experiment.results
    }
    result = replace(experiment, generator_descriptions=descriptions)

    output = _render(result)

    for name, description in descriptions.items():
        section = output.split(f"Domain: {name}", 1)[1].split("Domain:", 1)[0]
        for line in description.splitlines():
            assert line in section
