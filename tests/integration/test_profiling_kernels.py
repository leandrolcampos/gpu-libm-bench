# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

import ctypes
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
    if DeviceContext.device_count() == 0:
        pytest.skip("No NVIDIA devices are visible to the CUDA driver.")

    with DeviceContext() as context:
        yield context


@pytest.fixture
def profiling_module(context: DeviceContext) -> DeviceModule:
    device_info = context.device_info
    target = CompilationTarget(
        device_info.backend, "nvptx64-nvidia-cuda", device_info.architecture
    )

    configured = os.environ.get("GPU_LIBM_BENCH_PROFILING_IMAGE")
    if configured is not None:
        path = Path(configured)
    else:
        image = resolve_device_image("llvm-libm", target=target)
        path = image.image_path.parent / "backends" / "profiling.cubin"

    return context.load_module(DeviceImage(path, target, "backends", (), None))


@pytest.mark.parametrize(("release", "timed_out"), [(1, 0), (0, 1)])
def test_block_stream_release_and_timeout(
    context: DeviceContext,
    profiling_module: DeviceModule,
    release: int,
    timed_out: int,
) -> None:
    release_buffer = context.create_buffer(1, dtype=np.uint32)
    timeout_buffer = context.create_buffer(1, dtype=np.uint32)
    result = np.array([99], dtype=np.uint32)

    context.copy_to_device(release_buffer, np.array([release], dtype=np.uint32))
    context.copy_to_device(timeout_buffer, result)

    kernel = profiling_module.get_kernel(
        "blockStream",
        spec=KernelSpec(
            (
                KernelArgument("release", "uint32", True),
                KernelArgument("timed_out", "uint32", True),
                KernelArgument("timeout_ticks", "uint64", False),
            )
        ),
    )

    context.launch(
        kernel,
        release_buffer,
        timeout_buffer,
        np.uint64(100_000),
        config=LaunchConfig(block=(2, 2, 2)),
    )
    context.copy_to_host(result, timeout_buffer)

    assert result[0] == timed_out


def test_clock_counters_read_from_sm_zero(
    context: DeviceContext, profiling_module: DeviceModule
) -> None:
    driver = ctypes.CDLL("libcuda.so.1")
    device = ctypes.c_int()
    sm_count = ctypes.c_int()
    assert driver.cuDeviceGet(ctypes.byref(device), context.device_id) == 0
    assert driver.cuDeviceGetAttribute(ctypes.byref(sm_count), 16, device) == 0

    arrays = (np.zeros(1, dtype=np.uint64), np.zeros(1, dtype=np.uint64))
    buffers = tuple(context.create_buffer(1, dtype=np.uint64) for _ in arrays)

    for buffer, array in zip(buffers, arrays, strict=True):
        context.copy_to_device(buffer, array)

    kernel = profiling_module.get_kernel(
        "readClockCounters",
        spec=KernelSpec(
            (
                KernelArgument("steady_counter", "uint64", True),
                KernelArgument("cycle_counter", "uint64", True),
            )
        ),
    )

    context.launch(kernel, *buffers, config=LaunchConfig(grid=(sm_count.value, 1, 1)))

    for buffer, array in zip(buffers, arrays, strict=True):
        context.copy_to_host(array, buffer)

    for array in arrays:
        assert array[0] > 0
