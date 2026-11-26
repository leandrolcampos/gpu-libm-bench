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

"""Tests for independent-call throughput measurement, analysis, and launch planning."""

from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest

from gpu_libm_bench.benchmarks import throughput
from gpu_libm_bench.configs import (
    FunctionConfig,
    ThroughputSearchConfig,
    load_function_signature_config,
)
from gpu_libm_bench.devices import (
    DeviceContext,
    DeviceInfo,
    DeviceKernel,
    KernelMetric,
    KernelProfile,
    LaunchConfig,
    LaunchLimits,
    TimerInfo,
)
from gpu_libm_bench.inputs import InputBatch
from gpu_libm_bench.kernels import KernelKind, KernelRequest, KernelSpec


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
    function = FunctionConfig("expf", load_function_signature_config("expf"), {})
    kernel = DeviceKernel(
        native, KernelSpec.from_config(function, kind=KernelKind.THROUGHPUT)
    )
    return (
        context,
        kernel,
        KernelRequest("llvm-libm", "expf", KernelKind.THROUGHPUT, unroll_factor=4),
    )


@pytest.fixture
def search() -> ThroughputSearchConfig:
    return ThroughputSearchConfig(
        max_candidates=64,
        batch_volume_to_l2_ratios=[0.75, 2.25],
        fallback_batch_volumes_mib=[32, 64],
        warps_per_block=[4, 8, 16, 32],
        waves_per_grid=[1, 2],
    )


# Measurement and analysis


@pytest.mark.parametrize("pass_count", (0, 2**32))
def test_benchmark_rejects_pass_count_outside_positive_uint32_range(
    throughput_device: tuple[Mock, DeviceKernel, KernelRequest], pass_count: int
) -> None:
    context, kernel, request = throughput_device
    launch = throughput.try_plan_throughput_launch(
        context.device_info,
        kernel,
        request,
        batch_size=1024,
        threads_per_block=256,
        blocks_in_grid=1,
    )
    assert launch is not None
    with pytest.raises(ValueError, match="pass_count must"):
        throughput.ThroughputBenchmark(
            context, kernel, launch, sample_size=2, pass_count=pass_count, warmup_runs=0
        )


def test_measure_retries_throttled_batches_and_analyze_uses_accepted_gpu_time(
    throughput_device: tuple[Mock, DeviceKernel, KernelRequest],
) -> None:
    context, kernel, request = throughput_device
    pass_count = 64
    launch = throughput.try_plan_throughput_launch(
        context.device_info,
        kernel,
        request,
        batch_size=1024,
        threads_per_block=256,
        blocks_in_grid=1,
    )
    assert launch is not None
    benchmark = throughput.ThroughputBenchmark(
        context,
        kernel,
        launch,
        sample_size=2,
        pass_count=pass_count,
        warmup_runs=1,
        min_clock_ratio=0.75,
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
    assert [attempt.discard_reason for attempt in measurement.attempts] == [
        "clock below threshold",
        None,
        None,
    ]
    assert context.launch_and_profile.call_args.kwargs["metrics"] == {
        KernelMetric.DURATION,
        KernelMetric.CLOCK_FREQUENCY,
    }
    assert context.launch_and_profile.call_args.args[1:3] == (
        np.uint32(pass_count),
        np.uint32(0),
    )

    result = throughput.analyze(measurement)
    assert result is not None
    assert result.aggregate == pytest.approx(2048 * pass_count / (4 * 1e6))
    assert result.min == pytest.approx(1024 * pass_count / (3 * 1e6))
    assert result.max == pytest.approx(1024 * pass_count / 1e6)
    assert result.stdev == pytest.approx((result.max - result.min) / np.sqrt(2))


def test_measure_keeps_partial_sample_and_analyze_rejects_it_on_timeout(
    throughput_device: tuple[Mock, DeviceKernel, KernelRequest],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, kernel, request = throughput_device
    launch = throughput.try_plan_throughput_launch(
        context.device_info,
        kernel,
        request,
        batch_size=1024,
        threads_per_block=256,
        blocks_in_grid=1,
    )
    assert launch is not None
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


# Launch geometry


@pytest.mark.parametrize(
    ("occupancy", "batch_size", "expected"),
    [
        ((2, 4), 2048, (256, 2)),
        ((2, 4), 1023, (128, 1)),
        ((2, 4), 511, None),
        ((1, 3), 8192, (128, 3)),
    ],
)
def test_launch_planner_prioritizes_residency_and_checks_batch_capacity(
    throughput_device: tuple[Mock, DeviceKernel, KernelRequest],
    search: ThroughputSearchConfig,
    monkeypatch: pytest.MonkeyPatch,
    occupancy: tuple[int, int],
    batch_size: int,
    expected: tuple[int, int] | None,
) -> None:
    context, kernel, request = throughput_device
    monkeypatch.setattr(
        kernel, "max_active_blocks_per_multiprocessor", Mock(side_effect=occupancy)
    )

    device_info = context.device_info
    planner = throughput._LaunchPlanner(
        device_info, kernel, request, warps_per_block=search.warps_per_block
    )
    budget = throughput._LaunchBufferBudget.from_device(kernel.spec, device_info)
    launches, exclusions = planner.plan_launches(
        (batch_size,), budget, waves_per_grid=search.waves_per_grid
    )
    if expected is None:
        assert not launches
        assert exclusions
        assert all(
            item.reason.endswith("too few input tuples for a full group per thread")
            for item in exclusions
        )
    else:
        block_size, grid_size = expected
        assert launches[0].config == LaunchConfig(
            grid_dim=(grid_size, 1, 1), block_dim=(block_size, 1, 1)
        )


def test_try_increase_throughput_batch_caps_growth_at_input_and_checksum_budget(
    throughput_device: tuple[Mock, DeviceKernel, KernelRequest],
    search: ThroughputSearchConfig,
) -> None:
    context, kernel, request = throughput_device
    native = kernel._native
    native.info["max_threads_per_block"] = 512
    cache_bytes = 256 << 20
    device_info = replace(
        context.device_info,
        l2_cache_size_bytes=cache_bytes,
        total_memory_bytes=8 << 30,
    )

    plan = throughput.plan_throughput_search(
        device_info, {request: kernel}, search=replace(search, max_candidates=80)
    )

    largest = max(plan.candidates, key=lambda launch: launch.batch_size)
    larger_batch = throughput.try_increase_throughput_batch(
        largest, device_info, kernel, batch_volume_growth_ratio=0.5
    )
    assert larger_batch is not None and larger_batch.batch_size > largest.batch_size
    assert larger_batch.config == largest.config
    assert (larger_batch.batch_size + larger_batch.thread_count) * 4 == 512 << 20


# Search planning


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_candidates", 3),
        ("batch_volume_to_l2_ratios", [float("inf")]),
        ("fallback_batch_volumes_mib", [0]),
        ("warps_per_block", [8, 8]),
        ("waves_per_grid", []),
    ],
)
def test_plan_throughput_search_rejects_invalid_parameters(
    throughput_device: tuple[Mock, DeviceKernel, KernelRequest],
    search: ThroughputSearchConfig,
    field: str,
    value: int | list[int] | list[float],
) -> None:
    context, kernel, request = throughput_device
    with pytest.raises(ValueError, match=f"{field} must"):
        throughput.plan_throughput_search(
            context.device_info,
            {request: kernel},
            search=replace(search, **{field: value}),
        )


def test_plan_throughput_search_falls_back_to_largest_resident_block(
    throughput_device: tuple[Mock, DeviceKernel, KernelRequest],
    search: ThroughputSearchConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, kernel, request = throughput_device
    monkeypatch.setattr(
        kernel,
        "max_active_blocks_per_multiprocessor",
        lambda *, threads_per_block: 2 if threads_per_block in (224, 192) else 0,
    )

    device_info = replace(context.device_info, total_memory_bytes=1 << 30)
    plan = throughput.plan_throughput_search(
        device_info, {request: kernel}, search=replace(search, max_candidates=4)
    )

    assert plan.candidates
    assert {launch.config.block_dim[0] for launch in plan.candidates} == {224}


def test_plan_throughput_search_interleaves_unroll_factors_within_budget(
    throughput_device: tuple[Mock, DeviceKernel, KernelRequest],
    search: ThroughputSearchConfig,
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

    plan = throughput.plan_throughput_search(
        device_info, kernels, search=replace(search, max_candidates=24)
    )
    short = throughput.plan_throughput_search(
        device_info,
        dict(reversed(tuple(kernels.items()))),
        search=replace(search, max_candidates=4),
    )

    assert len(plan.candidates) == 24
    assert short.candidates == plan.candidates[:4]
    assert [launch.request.unroll_factor for launch in short.candidates] == [1, 2, 4, 8]
    assert {launch.batch_size for launch in plan.eligible} == {9 << 20, 15 << 20}
    assert {
        (launch.config.block_dim[0], launch.config.grid_dim[0])
        for launch in plan.eligible
    } == {(256, 1), (128, 1), (256, 2), (128, 2)}
    assert len(plan.eligible) == len(set(plan.eligible))


def test_plan_throughput_search_preserves_fallback_volume_order_without_l2_size(
    throughput_device: tuple[Mock, DeviceKernel, KernelRequest],
    search: ThroughputSearchConfig,
) -> None:
    context, kernel, request = throughput_device
    device_info = replace(context.device_info, total_memory_bytes=12 << 30)
    fallback_volumes_mib = [48, 16]

    plan = throughput.plan_throughput_search(
        device_info,
        {request: kernel},
        search=replace(
            search, max_candidates=4, fallback_batch_volumes_mib=fallback_volumes_mib
        ),
    )

    assert plan.batch_source == "fallback batch volumes (L2 cache size unavailable)"
    assert [launch.batch_size for launch in plan.candidates] == [
        volume * (1 << 20) // np.dtype(np.float32).itemsize
        for volume in fallback_volumes_mib
        for _ in range(2)
    ]
