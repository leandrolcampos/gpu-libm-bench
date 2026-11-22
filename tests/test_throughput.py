# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Tests for throughput launch planning and measurement."""

from dataclasses import replace
from unittest.mock import Mock, call

import numpy as np
import pytest

from gpu_libm_bench.benchmarks import throughput
from gpu_libm_bench.devices import (
    DeviceContext,
    DeviceInfo,
    DeviceKernel,
    KernelInfo,
    KernelMetric,
    KernelProfile,
    LaunchConfig,
    LaunchLimits,
    TimerInfo,
)
from gpu_libm_bench.inputs import InputBatch
from gpu_libm_bench.kernels import KernelArgument, KernelKind, KernelRequest, KernelSpec


@pytest.fixture
def throughput_device() -> tuple[Mock, DeviceKernel, KernelRequest]:
    context = Mock(spec=DeviceContext)
    context.device_info = DeviceInfo(
        "cuda",
        0,
        "test device",
        "sm_120",
        1,
        32,
        2_000_000_000,
        1 << 24,
        0,
        LaunchLimits((1024, 1, 1), (1024, 1, 1), 1024, 0),
        TimerInfo("test timer", 64, 10**9),
    )
    native = Mock()
    native.info = {
        "max_threads_per_block": 256,
        "registers_per_thread": 32,
        "static_shared_memory_bytes": 0,
        "local_memory_bytes_per_thread": 0,
    }
    native.max_active_blocks_per_multiprocessor.return_value = 1
    kernel = DeviceKernel(
        native,
        KernelSpec(
            (
                KernelArgument("x", "float32", True),
                KernelArgument("checksums", "uint32", True),
                KernelArgument("count", "uint64", False),
            )
        ),
    )
    return (
        context,
        kernel,
        KernelRequest("llvm-libm", "expf", KernelKind.THROUGHPUT, unroll_factor=4),
    )


def test_search_covers_factors_and_geometries_with_a_stable_budget(
    throughput_device: tuple[Mock, DeviceKernel, KernelRequest],
) -> None:
    context, kernel, request = throughput_device
    device_info = replace(
        context.device_info,
        total_memory_bytes=1 << 30,
        l2_cache_size_bytes=48 << 20,
        launch_limits=replace(
            context.device_info.launch_limits, max_grid_dim=(2, 1, 1)
        ),
    )
    kernels = {
        replace(request, unroll_factor=factor): kernel for factor in (8, 4, 2, 1)
    }

    plan = throughput.plan_throughput_search(device_info, kernels, max_candidates=32)
    short = throughput.plan_throughput_search(
        device_info, dict(reversed(tuple(kernels.items()))), max_candidates=4
    )

    assert len(plan.candidates) == 32
    assert short.candidates == plan.candidates[:4]
    assert [launch.request.unroll_factor for launch in short.candidates] == [1, 2, 4, 8]
    assert all(launch.batch_size == 1 << 23 for launch in short.candidates)
    assert {launch.batch_size for launch in plan.candidates} == {
        4 << 20,
        8 << 20,
        12 << 20,
    }
    assert {
        (launch.config.block_dim[0], launch.config.grid_dim[0])
        for launch in plan.candidates
        if launch.request.unroll_factor == 4 and launch.batch_size == 1 << 23
    } >= {(256, 2), (128, 2), (256, 1)}
    assert len(plan.eligible) == len(set(plan.eligible))
    assert len(plan.eligible) > len(plan.candidates)
    assert any(
        "input and checksum buffers exceed" in item.reason for item in plan.exclusions
    )


def test_search_uses_fixed_batch_sizes_on_devices_without_l2(
    throughput_device: tuple[Mock, DeviceKernel, KernelRequest],
) -> None:
    context, kernel, request = throughput_device
    device_info = replace(context.device_info, total_memory_bytes=12 << 30)

    plan = throughput.plan_throughput_search(
        device_info, {request: kernel}, max_candidates=4
    )

    assert plan.batch_source == "device has no L2 cache"
    assert [launch.batch_size for launch in plan.candidates] == [
        8 << 20,
        4 << 20,
        1 << 20,
        16 << 20,
    ]


def test_plan_falls_back_when_primary_block_sizes_have_no_residency(
    throughput_device: tuple[Mock, DeviceKernel, KernelRequest],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, kernel, request = throughput_device
    occupancy = Mock(side_effect=[0, 0, 2])
    monkeypatch.setattr(kernel, "max_active_blocks_per_multiprocessor", occupancy)

    launch = throughput.try_plan_throughput_launch(
        context.device_info, kernel, request, batch_size=2048
    )
    assert launch == throughput.ThroughputPlannedLaunch(
        request,
        KernelInfo(256, 32, 0, 0),
        LaunchConfig(grid_dim=(2, 1, 1), block_dim=(224, 1, 1)),
        2,
        2048,
    )
    assert occupancy.call_args_list == [
        call(threads_per_block=256),
        call(threads_per_block=128),
        call(threads_per_block=224),
    ]


@pytest.mark.parametrize(
    ("occupancy", "batch_size", "expected"),
    [
        ((2, 4), 8192, (256, 8)),
        ((2, 4), 4096, (256, 4)),
        ((2, 4), 2048, (256, 2)),
        ((2, 4), 2047, None),
        ((1, 3), 8192, (128, 12)),
    ],
)
def test_plan_prioritizes_resident_warps_and_reduces_waves_for_small_batches(
    throughput_device: tuple[Mock, DeviceKernel, KernelRequest],
    monkeypatch: pytest.MonkeyPatch,
    occupancy: tuple[int, int],
    batch_size: int,
    expected: tuple[int, int] | None,
) -> None:
    context, kernel, request = throughput_device
    monkeypatch.setattr(
        kernel, "max_active_blocks_per_multiprocessor", Mock(side_effect=occupancy)
    )

    launch = throughput.try_plan_throughput_launch(
        context.device_info, kernel, request, batch_size=batch_size
    )
    if expected is None:
        assert isinstance(launch, throughput.ThroughputExcludedLaunch)
        assert launch.reason == (
            f"batch_size={batch_size} cannot give each thread 4 evaluations "
            "at any candidate launch geometry"
        )
    else:
        assert isinstance(launch, throughput.ThroughputPlannedLaunch)
        block_size, grid_size = expected
        assert launch.config == LaunchConfig(
            grid_dim=(grid_size, 1, 1), block_dim=(block_size, 1, 1)
        )


def test_measure_retries_the_same_batch_and_aggregates_accepted_gpu_time(
    throughput_device: tuple[Mock, DeviceKernel, KernelRequest],
) -> None:
    context, kernel, request = throughput_device
    launch = throughput.try_plan_throughput_launch(
        context.device_info, kernel, request, batch_size=1024
    )
    assert isinstance(launch, throughput.ThroughputPlannedLaunch)
    benchmark = throughput.ThroughputBenchmark(
        context, kernel, launch, sample_size=2, warmup_runs=1, min_clock_ratio=0.75
    )
    context.launch_and_profile.side_effect = [
        KernelProfile(clock_frequency_hz=1e9, duration_ms=7.0),
        KernelProfile(clock_frequency_hz=1.5e9, duration_ms=1.0),
        KernelProfile(clock_frequency_hz=2e9, duration_ms=3.0),
    ]
    first = np.ones(1024, dtype=np.float32)
    second = np.full(1024, 2, dtype=np.float32)
    batches = [InputBatch("central", {"x": array}) for array in (first, second)]

    measurement = benchmark.measure("central", batches)

    context.launch.assert_called_once()
    copies = [transfer.args[1] for transfer in context.copy_to_device.call_args_list]
    assert [array[0] for array in copies] == [1, 1, 1, 2]
    assert [attempt.batch_index for attempt in measurement.attempts] == [0, 0, 1]
    assert [attempt.discard_reason for attempt in measurement.attempts] == [
        "clock below threshold",
        None,
        None,
    ]
    assert context.launch_and_profile.call_args.kwargs["metrics"] == {
        KernelMetric.DURATION,
        KernelMetric.CLOCK_FREQUENCY,
    }
    result = throughput.analyze(measurement)
    assert result is not None
    assert result.mean == pytest.approx(2048 / (4 * 1e6))
    assert result.min == pytest.approx(1024 / (3 * 1e6))
    assert result.max == pytest.approx(1024 / 1e6)
    assert result.stdev == pytest.approx((result.max - result.min) / np.sqrt(2))


def test_timeout_preserves_attempts_without_producing_a_complete_result(
    throughput_device: tuple[Mock, DeviceKernel, KernelRequest],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, kernel, request = throughput_device
    launch = throughput.try_plan_throughput_launch(
        context.device_info, kernel, request, batch_size=1024
    )
    assert isinstance(launch, throughput.ThroughputPlannedLaunch)
    benchmark = throughput.ThroughputBenchmark(
        context, kernel, launch, sample_size=3, timeout_s=1, warmup_runs=0
    )
    elapsed = 0.0
    monkeypatch.setattr(throughput, "monotonic", lambda: elapsed)

    def profile(*args, **kwargs) -> KernelProfile:
        nonlocal elapsed
        elapsed += 0.6
        return KernelProfile(duration_ms=1)

    context.launch_and_profile.side_effect = profile
    batch = InputBatch("central", {"x": np.zeros(1024, dtype=np.float32)})
    measurement = benchmark.measure("central", (batch, batch))

    assert measurement.timed_out
    assert measurement.accepted_count == 1
    assert measurement.attempts[-1].discard_reason == "timeout"
    assert context.launch_and_profile.call_args.kwargs["metrics"] == {
        KernelMetric.DURATION
    }
    assert throughput.analyze(measurement) is None
