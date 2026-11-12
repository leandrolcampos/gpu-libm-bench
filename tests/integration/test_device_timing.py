# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for device timer readings."""

import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from gpu_libm_bench.compiler.images import resolve_device_image
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
    device_info = context.device_info
    target = CompilationTarget(
        device_info.backend, "nvptx64-nvidia-cuda", device_info.architecture
    )

    configured = os.environ.get("GPU_LIBM_BENCH_TIMER_IMAGE")
    if configured is not None:
        path = Path(configured)
    else:
        image = resolve_device_image("llvm-libm", target=target)
        path = image.image_path.parent / "tests" / "timer.cubin"

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
