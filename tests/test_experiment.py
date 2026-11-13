# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Tests for benchmark experiment orchestration."""

import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest
from hydra import compose, initialize_config_module

from gpu_libm_bench import inputs
from gpu_libm_bench.benchmarks import latency
from gpu_libm_bench.compiler.backends import BackendNotFoundError
from gpu_libm_bench.compiler.images import DeviceImageBuildInfo
from gpu_libm_bench.compiler.libraries import ResolvedLibrary
from gpu_libm_bench.compiler.targets import CompilationTarget
from gpu_libm_bench.compiler.toolchains import CUDAToolkit, LLVMToolchain
from gpu_libm_bench.configs import (
    DomainConfig,
    ExperimentConfig,
    LatencyConfig,
    LibraryConfig,
    RandomGeneratorConfig,
    register_config_schema,
    validate_experiment_config,
)
from gpu_libm_bench.devices import DeviceImage, DeviceInfo, TimerInfo
from gpu_libm_bench.experiment import (
    Experiment,
    ExperimentResult,
    LatencyResults,
    _latency,
)
from gpu_libm_bench.experiment import experiment as experiment_module


@pytest.fixture
def config() -> ExperimentConfig:
    register_config_schema()
    with initialize_config_module(config_module="gpu_libm_bench.configs.yaml"):
        composed = compose(
            config_name="experiment_config",
            overrides=[
                "function=expf",
                "metric=latency",
                "backend=cuda",
                "run.libraries=[llvm-libm]",
                "metric.sample_size=128",
            ],
        )

    return validate_experiment_config(composed)


@pytest.fixture
def experiment_result(config: ExperimentConfig) -> ExperimentResult:
    assert isinstance(config.metric, LatencyConfig)
    device_info = DeviceInfo(
        "cuda", 0, "test GPU", "sm_120", 1024, TimerInfo("timer", 64, 10**9)
    )
    measurement = latency.LatencyMeasurement(
        inputs.InputBatch(
            "central", {"x": np.ones(config.metric.sample_size, dtype=np.float32)}
        ),
        np.full(
            (config.metric.sample_size, config.metric.repetitions),
            config.metric.chain_length,
            dtype=np.uint64,
        ),
        device_info,
        pass_count=2,
        chain_length=config.metric.chain_length,
        warmup_runs=config.metric.warmup_runs,
    )

    target = CompilationTarget("cuda", config.backend.target_triple, "sm_120")
    image_build_info = DeviceImageBuildInfo(
        Path("/llvm/bin/clang++"),
        "clang version 24.0.0",
        "CUDA Toolkit",
        "13.4",
        Path("/cuda"),
        Path("/libm.bc"),
        DeviceImage(Path("/math.cubin"), target, "llvm-libm", ("expf",), None),
    )

    return ExperimentResult(
        config,
        device_info,
        {},
        Experiment(config).generator_descriptions,
        (LatencyResults(image_build_info, (latency.analyze(measurement),)),),
    )


@pytest.fixture
def execution(
    experiment_result: ExperimentResult, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    image_build_info = experiment_result.results[0].image_build_info
    context = MagicMock()
    context.__enter__.return_value = context
    context.device_info = experiment_result.device_info

    llvm = LLVMToolchain(
        Path("/llvm"),
        Path("/llvm/clang++"),
        Path("/llvm/llvm-dis"),
        Path("/llvm/llvm-nm"),
        (),
        "test LLVM",
    )
    sdk = CUDAToolkit(Path("/cuda"), Path("/cuda/bin/ptxas"), "13.4")

    def build(*, resolved_library, target, build_directory, **kwargs):
        return replace(
            image_build_info.image,
            image_path=build_directory / "math.cubin",
            library=resolved_library.library,
            target=target,
        )

    benchmarks = []

    def benchmark(context, kernel, *, chain_length, repetitions, warmup_runs):
        value = Mock()
        value.measure.side_effect = lambda batch: latency.LatencyMeasurement(
            batch,
            np.full((batch.size, repetitions), chain_length, dtype=np.uint64),
            context.device_info,
            pass_count=2,
            chain_length=chain_length,
            warmup_runs=warmup_runs,
        )
        benchmarks.append(value)
        return value

    monkeypatch.setattr(experiment_module, "resolve_sdk", Mock(return_value=sdk))
    monkeypatch.setattr(experiment_module, "resolve_llvm", Mock(return_value=llvm))

    monkeypatch.setattr(experiment_module, "get_backend", Mock())
    monkeypatch.setattr(experiment_module, "build_backend", Mock())

    monkeypatch.setattr(experiment_module, "DeviceContext", Mock(return_value=context))
    monkeypatch.setattr(
        experiment_module,
        "resolve_target",
        Mock(return_value=image_build_info.image.target),
    )

    monkeypatch.setattr(
        experiment_module.paths, "build_root", lambda: tmp_path / "build"
    )
    monkeypatch.setattr(
        experiment_module._libraries,
        "select_libraries",
        Mock(
            return_value=(ResolvedLibrary("llvm-libm", image_build_info.bitcode_path),)
        ),
    )
    monkeypatch.setattr(experiment_module, "build_kernels", Mock(side_effect=build))
    monkeypatch.setattr(
        experiment_module,
        "read_device_image_build_info",
        Mock(side_effect=lambda image: replace(image_build_info, image=image)),
    )
    monkeypatch.setattr(_latency, "LatencyBenchmark", Mock(side_effect=benchmark))

    return context, benchmarks


def test_run_generates_configured_domain_inputs(
    config: ExperimentConfig, execution
) -> None:
    config = replace(
        config,
        function=replace(
            config.function,
            domains={
                **config.function.domains,
                "wide": DomainConfig(
                    generator=RandomGeneratorConfig(
                        name="random",
                        arguments={
                            "x": {"distribution": "uniform", "low": 20, "high": 30}
                        },
                    )
                ),
            },
        ),
    )
    assert isinstance(config.metric, LatencyConfig)

    result = Experiment(config).run()
    batches = [item.measurement.inputs for item in result.results[0].results]

    assert [batch.domain for batch in batches] == ["central", "wide"]

    for batch in batches:
        assert set(batch.arguments) == {"x"}
        samples = batch.arguments["x"]

        generator = config.function.domains[batch.domain].generator
        assert isinstance(generator, RandomGeneratorConfig)
        bounds = generator.arguments["x"]

        assert samples.dtype == np.dtype(np.float32)
        assert samples.shape == (config.metric.sample_size,)
        assert samples.flags.c_contiguous
        assert np.all(samples >= bounds["low"])
        assert np.all(samples <= bounds["high"])


@pytest.mark.parametrize("domains", [None, ["wide"], ["wide", "central"]])
def test_selected_domain_inputs_are_reproducible(
    config: ExperimentConfig, domains: list[str] | None, execution
) -> None:
    config = replace(
        config,
        run=replace(config.run, domains=domains),
        metric=replace(config.metric, sample_size=32),
    )
    experiment = Experiment(config)
    first = experiment.run().results[0]
    expected_domains = list(config.function.domains) if domains is None else domains
    assert list(first) == expected_domains

    for repeated in (experiment.run(), Experiment(config).run()):
        for domain, result in repeated.results[0].items():
            np.testing.assert_array_equal(
                result.measurement.inputs.arguments["x"].view(np.uint32),
                first[domain].measurement.inputs.arguments["x"].view(np.uint32),
            )

    changed_seed = replace(config, run=replace(config.run, seed=123))
    changed = Experiment(changed_seed).run().results[0]
    domain = next(iter(first))

    assert not np.array_equal(
        changed[domain].measurement.inputs.arguments["x"],
        first[domain].measurement.inputs.arguments["x"],
    )


def test_inputs_are_reproducible_across_processes(tmp_path: Path) -> None:
    script = """
from hydra import compose, initialize_config_module
from gpu_libm_bench.configs import register_config_schema, validate_experiment_config
from gpu_libm_bench.experiment import Experiment

register_config_schema()
with initialize_config_module(config_module="gpu_libm_bench.configs.yaml"):
    config = compose(config_name="experiment_config", overrides=[
        "function=expf", "metric=latency", "backend=cuda", "run.libraries=[llvm-libm]",
    ])

generator = next(
    iter(Experiment(validate_experiment_config(config))._input_generators.values())
)
batch = next(generator.batches(batch_size=32))
print(batch.arguments["x"].tobytes().hex())
"""
    outputs = []
    for _ in range(2):
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        outputs.append(result.stdout)

    assert outputs[0] == outputs[1]


def test_identical_domains_use_distinct_streams(
    config: ExperimentConfig, execution
) -> None:
    paired = replace(
        config,
        function=replace(
            config.function,
            domains={
                "left": config.function.domains["central"],
                "right": config.function.domains["central"],
            },
        ),
    )
    results = Experiment(paired).run().results[0]

    assert not np.array_equal(
        results["left"].measurement.inputs.arguments["x"],
        results["right"].measurement.inputs.arguments["x"],
    )


def test_input_sampling_does_not_consume_ordering_stream(
    config: ExperimentConfig,
    execution,
) -> None:
    expected_ordering = Experiment(config).ordering_rng.random(16)
    experiment = Experiment(config)

    experiment.run()

    np.testing.assert_array_equal(experiment.ordering_rng.random(16), expected_ordering)


def test_run_collects_comparable_results_for_each_library(
    config: ExperimentConfig, execution, tmp_path: Path
) -> None:
    context, benchmarks = execution
    config = replace(
        config,
        libraries={
            **config.libraries,
            "other": replace(config.libraries["llvm-libm"], name="other"),
        },
        run=replace(config.run, libraries=["other", "llvm-libm"]),
    )
    experiment_module._libraries.select_libraries.return_value = (
        ResolvedLibrary("other", Path("/other.bc")),
        ResolvedLibrary("llvm-libm", Path("/libm.bc")),
    )

    result = Experiment(config).run()

    assert [
        call.kwargs["build_directory"]
        for call in experiment_module.build_kernels.call_args_list
    ] == [
        tmp_path / "build/kernels/cuda/sm_120" / library
        for library in ("other", "llvm-libm")
    ]

    assert [library.image_build_info.image.library for library in result.results] == [
        "other",
        "llvm-libm",
    ]

    first = [call.args[0] for call in benchmarks[0].measure.call_args_list]
    second = [call.args[0] for call in benchmarks[1].measure.call_args_list]
    assert [batch.domain for batch in first] == ["central", "wide"]
    assert [batch.domain for batch in second] == ["central", "wide"]

    for left, right in zip(first, second, strict=True):
        for name in left.arguments:
            np.testing.assert_array_equal(left.arguments[name], right.arguments[name])

    assert result.device_info == context.device_info
    assert result.config.backend.architecture == "sm_120"
    assert config.backend.architecture is None
    assert result.environment["NumPy"] == np.__version__

    experiment_module.build_backend.assert_not_called()
    context.__exit__.assert_called_once()


def test_run_builds_a_missing_backend(config: ExperimentConfig, execution) -> None:
    experiment_module.get_backend.side_effect = BackendNotFoundError(
        "extension is missing"
    )

    Experiment(config).run()

    experiment_module.build_backend.assert_called_once_with(
        "cuda",
        sdk=experiment_module.resolve_sdk.return_value,
    )


@pytest.mark.parametrize(
    "operation",
    ["build_kernels", "measure"],
)
def test_run_propagates_failures_and_closes_the_context(
    config: ExperimentConfig, execution, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    context, _ = execution
    error = RuntimeError(f"{operation} failed")
    if operation == "measure":
        benchmark = Mock()
        benchmark.measure.side_effect = error
        monkeypatch.setattr(_latency, "LatencyBenchmark", Mock(return_value=benchmark))
    else:
        monkeypatch.setattr(experiment_module, operation, Mock(side_effect=error))

    with pytest.raises(RuntimeError) as caught:
        Experiment(config).run()

    assert caught.value is error
    context.__exit__.assert_called_once()


def test_latency_results_use_unique_measurement_domains(
    experiment_result: ExperimentResult,
) -> None:
    library = experiment_result.results[0]
    central = library["central"]
    measurement = central.measurement
    wide = latency.analyze(
        replace(measurement, inputs=replace(measurement.inputs, domain="wide"))
    )
    results = replace(library, results=(wide, central))

    assert list(results) == ["wide", "central"]

    with pytest.raises(ValueError, match="at most one result per domain"):
        replace(library, results=(central, central))


@pytest.mark.parametrize(
    ("mismatch", "message"),
    [
        ("backend", "device backend does not match"),
        ("duplicate_library", "at most one result per library"),
        ("unknown_library", "unknown library"),
        ("image_target", "device image target does not match"),
        ("metric", "unsupported experiment metric"),
        ("measurement_device", "measurement device does not match device_info"),
        ("unknown_domain", "unknown domain"),
        ("missing_description", "has no generator description"),
    ],
)
def test_inconsistent_experiment_result_is_rejected_at_construction(
    experiment_result: ExperimentResult, mismatch: str, message: str
) -> None:
    changes = {}
    library = experiment_result.results[0]

    if mismatch == "backend":
        changes["device_info"] = replace(experiment_result.device_info, backend="rocm")
    elif mismatch in ("measurement_device", "unknown_domain"):
        result = library["central"]
        measurement = result.measurement

        if mismatch == "measurement_device":
            measurement = replace(
                measurement,
                device_info=replace(experiment_result.device_info, id=1),
            )
        else:
            measurement = replace(
                measurement, inputs=replace(measurement.inputs, domain="unknown")
            )
        changes["results"] = (
            replace(
                library,
                results=(replace(result, measurement=measurement),),
            ),
        )
    elif mismatch == "missing_description":
        changes["generator_descriptions"] = {}
    elif mismatch == "metric":
        changes["config"] = replace(
            experiment_result.config,
            metric=replace(experiment_result.config.metric, name="unsupported"),
        )
    elif mismatch == "duplicate_library":
        changes["results"] = (library, library)
    else:
        image_build_info = library.image_build_info
        image = image_build_info.image
        if mismatch == "unknown_library":
            image = replace(image, library="unknown")
        else:
            image = replace(image, target=replace(image.target, architecture="sm_90"))
        changes["results"] = (
            replace(library, image_build_info=replace(image_build_info, image=image)),
        )

    with pytest.raises(ValueError, match=message):
        replace(experiment_result, **changes)


def test_latency_libraries_require_matching_domains(
    experiment_result: ExperimentResult,
) -> None:
    first = experiment_result.results[0]
    image_build_info = first.image_build_info
    second = replace(
        first,
        image_build_info=replace(
            image_build_info, image=replace(image_build_info.image, library="other")
        ),
        results=(),
    )
    config = replace(
        experiment_result.config,
        libraries={
            **experiment_result.config.libraries,
            "other": LibraryConfig("other", {}),
        },
    )

    with pytest.raises(ValueError, match="same domains"):
        replace(experiment_result, config=config, results=(first, second))
