# Copyright 2026 Leandro A. Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for device timer readings and measured work."""

import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from gpu_libm_bench.compiler.targets import CompilationTarget
from gpu_libm_bench.devices import (
    DeviceContext,
    DeviceImage,
    DeviceModule,
    LaunchConfig,
)
from gpu_libm_bench.kernels import KernelArgument, KernelSpec

pytestmark = [pytest.mark.cuda, pytest.mark.execution]


@pytest.fixture
def context() -> Iterator[DeviceContext]:
    with DeviceContext() as value:
        yield value


@pytest.fixture
def timer_module(context: DeviceContext) -> DeviceModule:
    info = context.get_device_info(context.device_id, backend=context.backend)
    target = CompilationTarget(info.backend, "nvptx64-nvidia-cuda", info.architecture)
    build_root = Path(__file__).resolve().parents[2] / "build" / "kernels"
    path = Path(
        os.environ.get(
            "GPU_LIBM_BENCH_TIMER_IMAGE",
            build_root / target.backend / target.architecture / "tests" / "timer.cubin",
        )
    )

    return context.load_module(DeviceImage(path, target, "tests", (), None))


def test_timer_readings_advance_without_going_backwards(
    context: DeviceContext, timer_module: DeviceModule
) -> None:
    count = 4096
    readings = np.empty(2 * count, dtype=np.uint64)
    buffer = context.create_buffer(readings.size, dtype=readings.dtype)
    spec = KernelSpec(
        (
            KernelArgument("readings", "uint64", True),
            KernelArgument("count", "uint32", False),
        )
    )
    kernel = timer_module.get_kernel("timerReadings", spec=spec)

    context.launch(kernel, buffer, np.uint32(count), config=LaunchConfig())
    context.copy_to_host(readings, buffer)

    assert (readings[1:] >= readings[:-1]).all()
    assert readings[-1] > readings[0]


@pytest.mark.parametrize("iterations", [0, 1024])
def test_timer_measures_completed_dependent_work(
    context: DeviceContext, timer_module: DeviceModule, iterations: int
) -> None:
    count = 17
    seed = 7
    elapsed = np.empty(count, dtype=np.uint64)
    outputs = np.empty(count, dtype=np.uint32)
    durations = context.create_buffer(count, dtype=elapsed.dtype)
    results = context.create_buffer(count, dtype=outputs.dtype)
    spec = KernelSpec(
        (
            KernelArgument("elapsed", "uint64", True),
            KernelArgument("outputs", "uint32", True),
            KernelArgument("iterations", "uint32", False),
            KernelArgument("count", "uint32", False),
            KernelArgument("seed", "uint32", False),
            KernelArgument("mask", "uint32", False),
        )
    )
    kernel = timer_module.get_kernel("timerWork", spec=spec)

    context.launch(
        kernel,
        durations,
        results,
        np.uint32(iterations),
        np.uint32(count),
        np.uint32(seed),
        np.uint32(0),
        config=LaunchConfig(),
    )
    context.copy_to_host(elapsed, durations)
    context.copy_to_host(outputs, results)

    expected = seed
    for _ in range(iterations):
        expected = (expected * 1664525 + 1013904223) & 0xFFFFFFFF

    assert (outputs == expected).all()
    if iterations:
        assert (elapsed > 0).any()
