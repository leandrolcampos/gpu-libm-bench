# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Tests for benchmark experiment orchestration and result validation."""

import subprocess
import sys
from dataclasses import replace
from math import exp, log, sqrt
from pathlib import Path
from statistics import mean, stdev
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest
from hydra import compose, initialize_config_module
from rich.console import Console

from gpu_libm_bench import inputs
from gpu_libm_bench.benchmarks import latency, throughput
from gpu_libm_bench.compiler.backends import BackendModuleNotFoundError
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
    ThroughputConfig,
    register_config_schema,
    validate_experiment_config,
)
from gpu_libm_bench.devices import (
    DeviceImage,
    DeviceInfo,
    KernelInfo,
    LaunchLimits,
    TimerInfo,
)
from gpu_libm_bench.experiment import (
    Experiment,
    ExperimentResult,
    LatencyResults,
    ThroughputResults,
    _latency,
    _throughput,
)
from gpu_libm_bench.experiment import experiment as experiment_module
from gpu_libm_bench.kernels import KernelKind, KernelSpec
from gpu_libm_bench.reporting.terminal import print_result


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
                "metric.sample.size=128",
            ],
        )

    return validate_experiment_config(composed)


@pytest.fixture
def experiment_result(config: ExperimentConfig) -> ExperimentResult:
    assert isinstance(config.metric, LatencyConfig)
    device_info = DeviceInfo(
        "cuda",
        0,
        "test GPU",
        "sm_120",
        16,
        32,
        2_000_000_000,
        1024,
        0,
        LaunchLimits((2**31 - 1, 65_535, 65_535), (1024, 1024, 64), 1024, 48 * 1024),
        TimerInfo("timer", 64, 10**9),
    )
    measurement = latency.LatencyMeasurement(
        inputs=inputs.InputBatch(
            "central", {"x": np.ones(config.metric.sample.size, dtype=np.float32)}
        ),
        elapsed_ticks=np.full(
            config.metric.sample.size,
            config.metric.chain_length,
            dtype=np.uint64,
        ),
        device_info=device_info,
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
    backend_sdk = CUDAToolkit(Path("/cuda"), Path("/cuda/bin/ptxas"), "13.4")

    def build(*, resolved_library, target, build_directory, **kwargs):
        return replace(
            image_build_info.image,
            image_path=build_directory / "math.cubin",
            library=resolved_library.name,
            target=target,
        )

    benchmarks = []

    def benchmark(
        context,
        kernel,
        *,
        timeout_s,
        chain_length,
        warmup_runs,
        min_clock_ratio,
        initial_delay_s,
    ):
        value = Mock()
        value.measure.side_effect = lambda batch, *, record_clocks: (
            latency.LatencyMeasurement(
                inputs=batch,
                elapsed_ticks=np.full(batch.size, chain_length, dtype=np.uint64),
                device_info=context.device_info,
                pass_count=2,
                chain_length=chain_length,
                warmup_runs=warmup_runs,
            )
        )
        benchmarks.append(value)
        return value

    monkeypatch.setattr(
        experiment_module, "resolve_backend_sdk", Mock(return_value=backend_sdk)
    )
    monkeypatch.setattr(experiment_module, "resolve_llvm", Mock(return_value=llvm))

    monkeypatch.setattr(experiment_module, "get_backend_module", Mock())
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


# Result validation


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
def test_experiment_result_rejects_inconsistent_metadata_and_measurements(
    experiment_result: ExperimentResult, mismatch: str, message: str
) -> None:
    changes = {}
    library = experiment_result.results[0]

    if mismatch == "backend":
        changes["device_info"] = replace(experiment_result.device_info, backend="hip")
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


def test_latency_results_reject_duplicate_measurement_domains(
    experiment_result: ExperimentResult,
) -> None:
    library = experiment_result.results[0]
    central = library["central"]

    with pytest.raises(ValueError, match="at most one result per domain"):
        replace(library, results=(central, central))


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


# Input generation


def test_throughput_requires_at_least_two_confirmation_generators() -> None:
    register_config_schema()
    with initialize_config_module(config_module="gpu_libm_bench.configs.yaml"):
        composed = compose(
            config_name="experiment_config",
            overrides=[
                "function=expf",
                "metric=throughput",
                "backend=cuda",
                "metric.confirmation.repetitions=1",
            ],
        )

    config = validate_experiment_config(composed)
    with pytest.raises(ValueError, match=r"metric\.confirmation\.repetitions must"):
        Experiment(config)


def test_run_samples_inputs_within_each_domain_bounds(
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
        input_values = batch.arguments["x"]

        generator = config.function.domains[batch.domain].generator
        assert isinstance(generator, RandomGeneratorConfig)
        bounds = generator.arguments["x"]

        assert input_values.dtype == np.dtype(np.float32)
        assert input_values.shape == (config.metric.sample.size,)
        assert input_values.flags.c_contiguous
        assert np.all(input_values >= bounds["low"])
        assert np.all(input_values <= bounds["high"])


@pytest.mark.parametrize("domains", [None, ["wide"], ["wide", "central"]])
def test_selected_domain_inputs_are_reproducible(
    config: ExperimentConfig, domains: list[str] | None, execution
) -> None:
    assert isinstance(config.metric, LatencyConfig)
    config = replace(
        config,
        run=replace(config.run, domains=domains),
        metric=replace(config.metric, sample=replace(config.metric.sample, size=32)),
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


def test_input_sampling_does_not_consume_ordering_stream(
    config: ExperimentConfig,
    execution,
) -> None:
    expected_ordering = Experiment(config).ordering_rng.random(16)
    experiment = Experiment(config)

    experiment.run()

    np.testing.assert_array_equal(experiment.ordering_rng.random(16), expected_ordering)


# Experiment execution


def test_run_builds_a_missing_backend(config: ExperimentConfig, execution) -> None:
    experiment_module.get_backend_module.side_effect = BackendModuleNotFoundError(
        "extension is missing"
    )

    Experiment(config).run()

    experiment_module.build_backend.assert_called_once_with(
        "cuda",
        backend_sdk=experiment_module.resolve_backend_sdk.return_value,
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


def test_run_measures_each_library_with_the_same_domain_inputs(
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
        for library in ("support", "other", "llvm-libm")
    ]
    support = experiment_module.build_kernels.call_args_list[0].kwargs
    assert support["resolved_library"] == ResolvedLibrary("support")
    assert support["target"] == experiment_module.resolve_target.return_value

    assert [library.image_build_info.image.library for library in result.results] == [
        "other",
        "llvm-libm",
    ]

    control, first_benchmark, second_benchmark = benchmarks
    control.estimate_chain_cost.assert_called_once_with()
    assert all(
        library.chain_cost is control.estimate_chain_cost.return_value
        for library in result.results
    )

    first = [call.args[0] for call in first_benchmark.measure.call_args_list]
    second = [call.args[0] for call in second_benchmark.measure.call_args_list]
    assert [batch.domain for batch in first] == ["central", "wide"]
    assert [batch.domain for batch in second] == ["central", "wide"]

    for left, right in zip(first, second, strict=True):
        for name in left.arguments:
            np.testing.assert_array_equal(left.arguments[name], right.arguments[name])

    assert result.device_info == context.device_info
    assert result.config.backend.architecture == "sm_120"
    assert config.backend.architecture is None

    experiment_module.build_backend.assert_not_called()
    context.__exit__.assert_called_once()


@pytest.mark.parametrize(
    ("return_type", "name"), [("_Float16", "idf16"), ("float", "idf"), ("double", "id")]
)
def test_latency_control_matches_return_type(return_type, name):
    identity = _latency._get_identity_config(return_type)
    assert identity.name == name
    assert identity.signature.return_type == return_type
    (argument,) = identity.signature.arguments
    assert (argument.type, argument.role) == (return_type, "input")


@pytest.mark.parametrize(
    ("outcome", "gain_threshold", "expected_evidence"),
    [
        ("decline", 0.05, "Relevant gains ruled out in tested contrasts"),
        ("gain", 0.05, "Relevant gain detected: larger batch"),
        ("gain", 0.25, "Relevant gains ruled out in tested contrasts"),
        ("custom", 0.05, "Relevant gain not ruled out: larger batch"),
        ("partial", 0.05, "Relevant gains ruled out in tested contrasts"),
        ("timeout", 0.05, "Incomplete larger batch contrast collection"),
        ("max_unroll", 0.05, "No viable parallelism contrast"),
        ("invalid_threshold", -0.01, None),
    ],
)
def test_throughput_selects_a_complete_search_winner_and_confirms_its_rate(
    execution,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
    gain_threshold: float,
    expected_evidence: str | None,
) -> None:
    custom_confirmation = outcome == "custom"
    confirmation_overrides = (
        [
            "metric.confirmation.repetitions=3",
            "metric.confirmation.confidence_level=0.9",
            "metric.confirmation.batch_volume_growth_ratio=0.25",
            "metric.confirmation.max_waves_per_grid=1",
        ]
        if custom_confirmation
        else []
    )
    with initialize_config_module(config_module="gpu_libm_bench.configs.yaml"):
        composed = compose(
            config_name="experiment_config",
            overrides=[
                "function=expf",
                "metric=throughput",
                "backend=cuda",
                "run.domains=[central]",
                "metric.sample.size=2",
                "metric.pass_count=1",
                "metric.search.max_candidates=4",
                "metric.search.batch_volume_to_l2_ratios=[0.75,3.0]",
                "metric.search.warps_per_block=[8]",
                "metric.search.waves_per_grid=[1,3]",
                f"metric.confirmation.relative_gain_threshold={gain_threshold}",
                *confirmation_overrides,
            ],
        )
    config = validate_experiment_config(composed)
    assert isinstance(config.metric, ThroughputConfig)
    context, _ = execution
    context.device_info = replace(
        context.device_info,
        multiprocessor_count=2,
        total_memory_bytes=1 << 30,
        l2_cache_size_bytes=48 << 10,
        launch_limits=(
            replace(context.device_info.launch_limits, max_grid_dim=(2, 1, 1))
            if outcome == "max_unroll"
            else context.device_info.launch_limits
        ),
    )
    kernel = context.load_module.return_value.get_kernel.return_value
    kernel.info = KernelInfo(256, 16, 0, 0)
    kernel.spec = KernelSpec.from_config(config.function, kind=KernelKind.THROUGHPUT)
    kernel.max_active_blocks_per_multiprocessor.return_value = 1
    monkeypatch.setattr(
        experiment_module._libraries,
        "select_libraries",
        Mock(
            side_effect=lambda configuration, **_: tuple(
                ResolvedLibrary(library, Path("/libm.bc"))
                for library in configuration.run.libraries or configuration.libraries
            )
        ),
    )
    measured_inputs = {library: [] for library in config.libraries}
    winning_unroll_factor = 8 if outcome == "max_unroll" else 2
    reference_batch_size = 9 << 10

    def benchmark(
        context, kernel, launch, *, sample_size, pass_count, warmup_runs, **kwargs
    ):
        def measure(domain, batches):
            observed = measured_inputs[launch.request.library]
            search = len(observed) < config.metric.search.max_candidates
            rate = (
                100.0 if launch.request.unroll_factor == winning_unroll_factor else 50.0
            )
            timed_out = (
                search and outcome == "partial" and launch.request.unroll_factor == 4
            )
            if timed_out:
                rate = 200.0
            if not search:
                timed_out = (
                    outcome == "timeout" and launch.batch_size > reference_batch_size
                )
                rate = 101.0
                if launch.batch_size > reference_batch_size:
                    if outcome == "decline":
                        rate = 75.0
                    elif outcome == "gain":
                        rate = 120.0
                    elif custom_confirmation:
                        repetition = sum(
                            size == launch.batch_size for size, _ in observed
                        )
                        rate = (90.0, 112.0)[repetition % 2]
            observed.append(
                (launch.batch_size, tuple(batch.arguments["x"] for batch in batches))
            )
            return throughput.ThroughputMeasurement(
                domain=domain,
                attempts=tuple(
                    throughput.ThroughputAttempt(
                        index, launch.batch_size / rate / 1e6, None, None
                    )
                    for index in range(1 if timed_out else sample_size)
                ),
                device_info=context.device_info,
                pass_count=pass_count,
                launch=launch,
                sample_size=sample_size,
                warmup_runs=warmup_runs,
                min_clock_frequency_hz=0,
                timed_out=timed_out,
            )

        return Mock(measure=measure)

    monkeypatch.setattr(_throughput._experiment, "ThroughputBenchmark", benchmark)

    if expected_evidence is None:
        with pytest.raises(
            ValueError, match=r"metric\.confirmation\.relative_gain_threshold must"
        ):
            Experiment(config).run()
        return

    experiment = Experiment(config).run()

    libraries = experiment.results_for(ThroughputResults)
    first_library, second_library = measured_inputs.values()
    for (size, inputs_a), (other_size, inputs_b) in zip(
        first_library, second_library, strict=True
    ):
        assert size == other_size
        np.testing.assert_array_equal(inputs_a, inputs_b)
    for _, batches in first_library[1:4]:
        np.testing.assert_array_equal(batches, first_library[0][1])
    for library in libraries:
        result = library["central"]
        assert result.search_winner is not None
        assert (
            result.search_winner.measurement.launch.request.unroll_factor
            == winning_unroll_factor
        )
        confirmation = result.confirmation
        assert confirmation is not None
        repetitions = 3 if custom_confirmation else 5
        assert len(confirmation.reference_measurements) == repetitions
        assert result.reported_result is not None
        assert (
            result.reported_result.measurement.launch
            == result.search_winner.measurement.launch
        )
        assert result.reported_result.aggregate == pytest.approx(101.0)
        assert result.reported_result.measurement.accepted_count == 2 * repetitions
        if custom_confirmation:
            batch_contrast, grid_contrast, *_ = confirmation.contrasts
            assert batch_contrast.launch is not None
            assert batch_contrast.launch.batch_size == reference_batch_size * 5 // 4
            assert grid_contrast.launch is None

            log_ratios = [
                log((90, 112)[index % 2] / 101) for index in range(repetitions)
            ]
            bounds = batch_contrast.ratio_bounds
            assert bounds is not None
            half_width = 1.885618083 * stdev(log_ratios) / sqrt(repetitions)
            assert bounds.lower == pytest.approx(exp(mean(log_ratios) - half_width))
            assert bounds.upper == pytest.approx(exp(mean(log_ratios) + half_width))

    console = Console(width=132)
    with console.capture() as captured:
        print_result(experiment, show_details=True, console=console)
    output = captured.get()
    assert f"Local stability: {expected_evidence}" in output
    if custom_confirmation:
        assert "one-sided 90% bounds" in output
        assert "Passes: 1" in output
        assert "Warmup runs: 10" in output
        assert (
            "Launches: 2 accepted launches per repetition (sample.size) "
            "× 3 repetitions (confirmation.repetitions)." in output
        )

        with console.capture() as captured:
            print_result(experiment, show_details=False, console=console)

        output = captured.get()
        assert "Aggregate" in output
        assert "Launches" in output
        assert "2 × 3" in output
        assert "Candidates" in output
        assert "Local stability" not in output

    if outcome == "decline":
        confirmation_inputs = first_library[4][1]
        assert not np.array_equal(confirmation_inputs, first_library[0][1])
        assert not np.array_equal(confirmation_inputs, first_library[8][1])
        np.testing.assert_array_equal(confirmation_inputs, first_library[6][1])
        library_name = next(iter(config.libraries))
        measured_inputs[library_name] = []
        isolated = replace(config, run=replace(config.run, libraries=[library_name]))
        Experiment(isolated).run()
        for (_, joint), (_, solo) in zip(
            first_library, measured_inputs[library_name], strict=True
        ):
            np.testing.assert_array_equal(joint, solo)
