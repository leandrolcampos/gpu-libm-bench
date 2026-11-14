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
from gpu_libm_bench.compiler.images import DeviceImageBuildInfo
from gpu_libm_bench.compiler.targets import CompilationTarget
from gpu_libm_bench.configs import (
    ArgumentConfig,
    BackendConfig,
    DomainConfig,
    ExperimentConfig,
    FunctionConfig,
    LatencyConfig,
    LibraryBitcodeConfig,
    LibraryConfig,
    LLVMConfig,
    RandomGeneratorConfig,
    RunConfig,
    SignatureConfig,
)
from gpu_libm_bench.devices import DeviceImage, DeviceInfo, TimerInfo
from gpu_libm_bench.experiment import ExperimentResult, LatencyResults
from gpu_libm_bench.reporting.terminal import print_result


@pytest.fixture
def experiment() -> ExperimentResult:
    target = CompilationTarget("cuda", "nvptx64-nvidia-cuda", "sm_120")
    device_info = DeviceInfo(
        "cuda",
        0,
        "test GPU",
        "sm_120",
        2_000_000_000,
        2**30,
        TimerInfo("%globaltimer", 64, 10**9),
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
        BackendConfig("cuda", target.target_triple, target.architecture, 0, "/cuda"),
        {
            "llvm-libm": LibraryConfig(
                "llvm-libm", {"cuda": LibraryBitcodeConfig("/libm.bc")}
            ),
            "variant": LibraryConfig(
                "variant", {"cuda": LibraryBitcodeConfig("/variant.bc")}
            ),
        },
        RunConfig(42, None),
        LLVMConfig("/llvm"),
    )

    image_build_info = DeviceImageBuildInfo(
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

    results = []
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

        latencies = (
            np.array([10.123456789, 14.123456789])
            if name == "central"
            else np.array([17.942, 19.542])
        )
        results.append(
            LatencyResult(
                measurement,
                latencies,
                float(np.mean(latencies)),
                stdev=float(np.std(latencies, ddof=1)),
                min=float(np.min(latencies)),
                max=float(np.max(latencies)),
            )
        )

    return ExperimentResult(
        config,
        device_info,
        {"Python": "3.12", "NumPy": "2.0"},
        {"central": "x: uniform(-10, 10)", "wide": "x: uniform(-80, 80)"},
        (
            LatencyResults(image_build_info, tuple(results)),
            LatencyResults(
                replace(
                    image_build_info,
                    bitcode_path=Path("/variant.bc"),
                    image=replace(
                        image_build_info.image,
                        library="variant",
                        image_path=Path("/variant.cubin"),
                    ),
                ),
                tuple(
                    replace(
                        result,
                        latencies=result.latencies + 10,
                        mean=result.mean + 10,
                        min=result.min + 10,
                        max=result.max + 10,
                    )
                    for result in reversed(results)
                ),
            ),
        ),
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


def test_report_presents_domain_statistics_without_details(
    experiment: ExperimentResult,
) -> None:
    output = _render(experiment)

    assert output.count("float expf(float x)") == 1
    assert output.count("Latency (ns/call)") == 1
    assert output.count("Domain: central") == output.count("Domain: wide") == 1
    assert "x: uniform(-10, 10)" in output

    assert "float expf(float x)\n\n  Latency (ns/call)" in output
    assert output.index("Latency (ns/call)") < output.index("CUDA — 0:")
    assert output.index("CUDA — 0:") < output.index("Domain: central")

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

    libraries = [
        line.split("│")[1].strip()
        for line in central.splitlines()
        if "llvm-libm" in line or "variant" in line
    ]
    assert libraries == ["llvm-libm", "variant"]
    assert "22.123" in central

    wide = output.split("Domain: wide", 1)[1]
    assert "18.742" in wide and "28.742" in wide

    assert "Configuration and Provenance" not in output
    assert "Compiler:" not in output and "Memory:" not in output
    assert "Timer:" not in output and "Repetitions:" not in output


def test_report_uses_recorded_generator_descriptions(
    experiment: ExperimentResult,
) -> None:
    descriptions = {
        name: f"x: recorded generator for {name}\nrange: recorded range for {name}"
        for name in experiment.results[0]
    }
    result = replace(experiment, generator_descriptions=descriptions)

    output = _render(result)

    for name, description in descriptions.items():
        section = output.split(f"Domain: {name}", 1)[1].split("Domain:", 1)[0]
        for line in description.splitlines():
            assert line in section


def test_domain_tables_preserve_widths_and_values(experiment: ExperimentResult) -> None:
    wide = replace(
        experiment.results[0]["wide"],
        mean=12345678.9,
        stdev=765432.1,
        min=10000000.0,
        max=15000000.0,
    )
    large = replace(
        experiment,
        results=(
            replace(
                experiment.results[0],
                results=(experiment.results[0]["central"], wide),
            ),
            experiment.results[1],
        ),
    )

    output = _render(large, width=40)

    headers = [line for line in output.splitlines() if "StDev" in line]
    assert len(headers) == 2
    assert headers[0] == headers[1]

    row = next(line for line in output.splitlines() if "1.2346e+07" in line)
    assert len(row) > 40
    assert "7.6543e+05" in row and "1.5e+07" in row


def test_report_includes_provenance_when_requested(
    experiment: ExperimentResult,
) -> None:
    output = _render(experiment, show_details=True)
    comparisons, provenance = output.split("Configuration and Provenance", 1)

    assert "Method:" not in comparisons and "Compiler:" not in comparisons
    assert "CUDA — 0: test GPU (sm_120)" in comparisons
    assert "Memory:" not in comparisons and "Timer:" not in comparisons

    assert provenance.count("Memory: 1,024.00 MiB") == 1
    assert provenance.count("Timer: %globaltimer, 64 bits, 1000000000 Hz") == 1
    assert provenance.count("Seed: 42") == 1
    assert provenance.count("Environment: Python=3.12; NumPy=2.0") == 1

    for line in (
        "Method: dependent-call chain (overhead included)",
        "Estimator: mean of per-sample medians",
        "Repetitions: 3",
        "Chain length: 512",
        "Passes: 2",
        "Warmup runs: 10",
    ):
        assert provenance.splitlines().count(f"  {line}") == 1

    assert "Domains: central, wide" in provenance

    methodology, builds = provenance.split("Library: llvm-libm", 1)
    assert "Method:" in methodology and "Method:" not in builds
    assert builds.count("Compiler:") == 2

    original, variant = provenance.split("Library: variant", 1)
    assert "/libm.bc" in original and "/variant.bc" not in original
    assert "/variant.bc" in variant and "/variant.cubin" in variant

    for metadata in (
        "clang version 24.0.0",
        "CUDA Toolkit 13.4",
        "/math.cubin",
        "-nogpulib",
    ):
        assert metadata in provenance

    for raw_value in ("1234567", "7654321", "987654321", "10.123456789"):
        assert raw_value not in output


def test_report_identifies_libraries_with_different_measurement_settings(
    experiment: ExperimentResult,
) -> None:
    variant = experiment.results[1]
    wide = variant["wide"]
    changed = replace(
        wide,
        measurement=replace(
            wide.measurement, elapsed_ticks=np.full((2, 5), 123, dtype=np.uint64)
        ),
    )
    result = replace(
        experiment,
        results=(
            experiment.results[0],
            replace(variant, results=(changed, variant["central"])),
        ),
    )

    output = _render(result, show_details=True)
    details = output.split("Configuration and Provenance", 1)[1].split("Library:", 1)[0]

    assert details.count("Domains: central") == 1

    original = details.split("Libraries: llvm-libm\n", 1)[1].split("Libraries:", 1)[0]
    variant = details.split("Libraries: variant\n", 1)[1].split("Libraries:", 1)[0]
    assert "Domains: wide\n" in original and "Repetitions: 3\n" in original
    assert "Domains: wide\n" in variant and "Repetitions: 5\n" in variant
