# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Tests for latency measurement and analysis."""

from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest

from gpu_libm_bench import inputs
from gpu_libm_bench.benchmarks import latency
from gpu_libm_bench.configs import (
    ArgumentConfig,
    FunctionConfig,
    SignatureConfig,
)
from gpu_libm_bench.devices import (
    DeviceBuffer,
    DeviceContext,
    DeviceInfo,
    DeviceKernel,
    KernelMetric,
    KernelProfile,
    TimerInfo,
)
from gpu_libm_bench.kernels import KernelKind, KernelSpec


@pytest.fixture
def latency_kernel() -> DeviceKernel:
    function = FunctionConfig(
        "example",
        SignatureConfig("float", [ArgumentConfig("argument", "float", "input")]),
        {},
    )
    return DeviceKernel(
        Mock(), KernelSpec.from_config(function, kind=KernelKind.LATENCY)
    )


@pytest.mark.parametrize(
    ("invalid_arguments", "error", "message"),
    [
        ({"chain_length": 0}, ValueError, "chain_length must be between"),
        ({"chain_length": 2**32}, ValueError, "chain_length must be between"),
        ({"chain_length": 1.5}, TypeError, "chain_length must be an integer"),
        ({"warmup_runs": -1}, ValueError, "warmup_runs must be nonnegative"),
        ({"warmup_runs": True}, TypeError, "warmup_runs must be an integer"),
    ],
)
def test_invalid_benchmark_arguments_are_rejected(
    latency_kernel: DeviceKernel, invalid_arguments, error, message
) -> None:
    arguments = dict(chain_length=1, warmup_runs=0)
    arguments.update(invalid_arguments)

    with pytest.raises(error, match=message):
        latency.LatencyBenchmark(Mock(spec=DeviceContext), latency_kernel, **arguments)


@pytest.mark.parametrize("warmup_runs", [0, 4])
@pytest.mark.parametrize(
    ("name", "type_name", "dtype", "binary"),
    [
        ("argument", "float", np.float32, False),
        ("output", "_Float16", np.float16, True),
        ("output", "double", np.float64, True),
    ],
)
def test_measurement_preserves_input_association_and_excludes_warmup(
    warmup_runs: int, name: str, type_name: str, dtype, binary: bool
) -> None:
    context = Mock(spec=DeviceContext)
    context.device_info = DeviceInfo(
        "cuda",
        3,
        "test device",
        "sm_120",
        16,
        2_000_000_000,
        1024,
        TimerInfo("test timer", 64, 10**9),
    )
    context.create_buffer.side_effect = lambda size, *, dtype: Mock(
        spec=DeviceBuffer, size=size, dtype=np.dtype(dtype)
    )
    observed_inputs = []
    observed_exponents = []

    def copy_to_device(destination, source):
        destination.data = source.copy()

    def launch(kernel, *arguments, config):
        kernel._validate_arguments(arguments)
        assert (arguments[0], arguments[1]) == (2, 7)
        np.testing.assert_array_equal(
            arguments[2].data, np.zeros(1, dtype=arguments[2].dtype)
        )

        value = int(arguments[5].data[0])
        observed_inputs.append(value)
        if binary:
            observed_exponents.append(int(arguments[6].data[0]))
        arguments[3].data = np.array(
            [100 * value + len(observed_inputs)], dtype=np.uint64
        )

    def copy_to_host(destination, source):
        np.copyto(destination, source.data)

    context.copy_to_device.side_effect = copy_to_device
    context.launch.side_effect = launch
    context.copy_to_host.side_effect = copy_to_host

    input_arguments = [ArgumentConfig(name, type_name, "input")]
    if binary:
        input_arguments.append(ArgumentConfig("exponent", "int", "input"))
    function = FunctionConfig(
        "example", SignatureConfig(type_name, input_arguments), {}
    )
    kernel = DeviceKernel(
        Mock(), KernelSpec.from_config(function, kind=KernelKind.LATENCY)
    )
    benchmark = latency.LatencyBenchmark(
        context,
        kernel,
        timeout_secs=60.0,
        chain_length=7,
        warmup_runs=warmup_runs,
    )

    values = np.array([11, 22, 33], dtype=dtype)
    exponents = np.array([-1, 2, 3], dtype=np.int32)
    arguments = {"exponent": exponents} if binary else {}
    arguments[name] = values
    batch = inputs.InputBatch("central", arguments)

    measurement = benchmark.measure(batch)

    expected = np.array([1101, 2202, 3303], dtype=np.uint64)
    np.testing.assert_array_equal(measurement.elapsed_ticks, expected + warmup_runs)
    assert measurement.device_info == context.device_info
    assert observed_inputs == [11, 22, 33, 11][:warmup_runs] + [11, 22, 33]
    if binary:
        assert observed_exponents == [-1, 2, 3, -1][:warmup_runs] + [-1, 2, 3]

    assert (
        measurement.chain_length,
        measurement.pass_count,
        measurement.warmup_runs,
    ) == (
        7,
        2,
        warmup_runs,
    )

    values.fill(-1)
    batch.arguments.clear()
    np.testing.assert_array_equal(measurement.inputs.arguments[name], [11, 22, 33])


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"argument": np.array([], dtype=np.float32)}, "nonempty one-dimensional"),
        ({"argument": np.zeros((2, 2), dtype=np.float32)}, "nonempty one-dimensional"),
        (
            {"argument": np.ones(2, dtype=np.float64)},
            "latency input 'argument' must have dtype float32",
        ),
        (
            {"other": np.ones(2, dtype=np.float32)},
            "must match the kernel input arguments",
        ),
    ],
)
def test_measurement_rejects_incompatible_inputs(
    latency_kernel: DeviceKernel, arguments, message: str
) -> None:
    context = Mock(spec=DeviceContext)
    benchmark = latency.LatencyBenchmark(
        context,
        latency_kernel,
        chain_length=1,
        warmup_runs=0,
    )

    with pytest.raises(ValueError, match=message):
        benchmark.measure(inputs.InputBatch("central", arguments))

    context.launch.assert_not_called()


def test_incompatible_elapsed_ticks_dtype_is_rejected(
    latency_kernel: DeviceKernel,
) -> None:
    spec = KernelSpec(
        tuple(
            replace(argument, dtype="uint32")
            if argument.name == "elapsed_ticks"
            else argument
            for argument in latency_kernel.spec.arguments
        )
    )
    kernel = DeviceKernel(Mock(), spec)

    with pytest.raises(ValueError, match="elapsed_ticks must have dtype uint64"):
        latency.LatencyBenchmark(
            Mock(spec=DeviceContext),
            kernel,
            chain_length=1,
            warmup_runs=0,
        )


def test_evaluation_kernel_is_rejected() -> None:
    function = FunctionConfig(
        "example",
        SignatureConfig("float", [ArgumentConfig("x", "float", "input")]),
        {},
    )
    kernel = DeviceKernel(
        Mock(), KernelSpec.from_config(function, kind=KernelKind.EVALUATION)
    )

    with pytest.raises(ValueError, match="kernel does not match the latency protocol"):
        latency.LatencyBenchmark(
            Mock(spec=DeviceContext),
            kernel,
            chain_length=1,
            warmup_runs=0,
        )


@pytest.mark.parametrize(
    ("exponents", "message"),
    [
        (np.ones(1, dtype=np.int32), "same number of elements"),
        (
            np.ones(2, dtype=np.float64),
            "latency input 'exponent' must have dtype int32",
        ),
    ],
)
def test_incompatible_secondary_input_is_rejected_before_allocation(
    exponents, message: str
) -> None:
    function = FunctionConfig(
        "example",
        SignatureConfig(
            "float",
            [
                ArgumentConfig("x", "float", "input"),
                ArgumentConfig("exponent", "int", "input"),
            ],
        ),
        {},
    )
    context = Mock(spec=DeviceContext)
    kernel = DeviceKernel(
        Mock(), KernelSpec.from_config(function, kind=KernelKind.LATENCY)
    )
    benchmark = latency.LatencyBenchmark(context, kernel, chain_length=1, warmup_runs=0)
    batch = inputs.InputBatch(
        "central",
        {"x": np.ones(2, dtype=np.float32), "exponent": exponents},
    )

    with pytest.raises(ValueError, match=message):
        benchmark.measure(batch)

    context.create_buffer.assert_not_called()


@pytest.mark.parametrize("scenario", ["filtered", "disabled", "timeout"])
def test_latency_throttling(
    scenario: str, latency_kernel: DeviceKernel, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = Mock(spec=DeviceContext)
    context.device_info = DeviceInfo(
        "cuda",
        0,
        "test GPU",
        "sm_120",
        16,
        2_000_000_000,
        1024,
        TimerInfo("timer", 64, 10**9),
    )
    context.create_buffer.side_effect = lambda size, *, dtype: Mock(spec=DeviceBuffer)
    context.copy_to_device.side_effect = lambda buffer, data: setattr(
        buffer, "data", data.copy()
    )
    context.copy_to_host.side_effect = lambda data, buffer: np.copyto(data, buffer.data)

    attempted_inputs = []
    delays = []
    now = 0.0

    def sleep(delay: float) -> None:
        nonlocal now
        delays.append(delay)
        now += delay

    def launch(kernel, *arguments, config):
        attempted_inputs.append(int(arguments[5].data[0]))
        arguments[3].data = np.array([len(attempted_inputs)], dtype=np.uint64)

    threshold = 1.5e9
    frequencies = iter(
        [threshold, 1.0e9]
        if scenario == "timeout"
        else [float("nan"), None, 1.0e9, 1.0e9, threshold, 2.0e9]
    )

    def profile(kernel, *arguments, config, metrics):
        assert metrics == {KernelMetric.CLOCK_FREQUENCY}
        launch(kernel, *arguments, config=config)
        return KernelProfile(clock_frequency_hz=next(frequencies))

    context.launch.side_effect = launch
    context.launch_and_profile.side_effect = profile
    monkeypatch.setattr(latency, "monotonic", lambda: now)
    monkeypatch.setattr(latency, "sleep", sleep)

    benchmark = latency.LatencyBenchmark(
        context,
        latency_kernel,
        timeout_secs=0.025 if scenario == "timeout" else 1.0,
        chain_length=1,
        warmup_runs=0,
        min_clock_ratio=0.0 if scenario == "disabled" else 0.75,
        initial_delay_secs=0.05,
    )
    batch = inputs.InputBatch("central", {"argument": np.array([11, 22], np.float32)})

    if scenario == "timeout":
        with pytest.raises(TimeoutError, match=r"central.*1/2.*input index 1 pending"):
            benchmark.measure(batch)
        assert attempted_inputs == [11, 22]
        assert delays == [0.025]
        return

    measurement = benchmark.measure(batch)
    if scenario == "disabled":
        context.launch_and_profile.assert_not_called()
        assert attempted_inputs == [11, 22]
        assert delays == []
        np.testing.assert_array_equal(measurement.elapsed_ticks, [1, 2])
    else:
        context.launch.assert_not_called()
        assert attempted_inputs == [11, 11, 11, 11, 11, 22]
        assert delays == pytest.approx([0.05, 0.05, 0.05, 0.125])
        np.testing.assert_array_equal(measurement.elapsed_ticks, [5, 6])


def test_analysis_summarizes_per_input_latencies() -> None:
    measurement = latency.LatencyMeasurement(
        inputs=inputs.InputBatch("central", {"x": np.ones(3, dtype=np.float32)}),
        elapsed_ticks=np.array([4, 20, 40], dtype=np.uint64),
        device_info=DeviceInfo(
            "cuda",
            0,
            "test device",
            "sm_120",
            16,
            2_000_000_000,
            1024,
            TimerInfo("test timer", 64, 500_000_000),
        ),
        chain_length=4,
        pass_count=2,
        warmup_runs=3,
    )
    result = latency.analyze(measurement)

    np.testing.assert_array_equal(result.latencies, [2, 10, 20])
    assert result.mean == pytest.approx(32 / 3)
    assert result.stdev == pytest.approx((244 / 3) ** 0.5)
    assert result.min == 2
    assert result.max == 20


def test_single_observation_has_no_sample_standard_deviation() -> None:
    measurement = latency.LatencyMeasurement(
        inputs.InputBatch("central", {"x": np.ones(1, dtype=np.float32)}),
        np.array([4], dtype=np.uint64),
        DeviceInfo(
            "cuda",
            0,
            "test GPU",
            "sm_120",
            16,
            2_000_000_000,
            1024,
            TimerInfo("timer", 64, 10**9),
        ),
        pass_count=2,
        chain_length=2,
        warmup_runs=0,
    )
    result = latency.analyze(measurement)

    assert result.stdev is None
    assert result.min == result.mean == result.max == 2
