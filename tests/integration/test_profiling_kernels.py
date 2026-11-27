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

"""Integration tests for stream blocking and clock counter kernels."""

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
    target = CompilationTarget(
        context.backend, "nvptx64-nvidia-cuda", context.device_info.architecture
    )
    configured = os.environ.get("GPU_LIBM_BENCH_PROFILING_IMAGE")
    image = resolve_device_image(
        "support", target=target, image_path=Path(configured) if configured else None
    )
    return context.load_module(image)


@pytest.mark.parametrize(("released", "timed_out"), [(1, 0), (0, 1)])
def test_block_stream_release_and_timeout(
    context: DeviceContext,
    profiling_module: DeviceModule,
    released: int,
    timed_out: int,
) -> None:
    released_buffer = context.create_buffer(1, dtype=np.uint32)
    timeout_buffer = context.create_buffer(1, dtype=np.uint32)
    result = np.array([99], dtype=np.uint32)
    context.copy_to_device(released_buffer, np.array([released], dtype=np.uint32))
    context.copy_to_device(timeout_buffer, result)

    kernel = profiling_module.get_kernel(
        "blockStream",
        spec=KernelSpec(
            (
                KernelArgument("released", "uint32", True),
                KernelArgument("timed_out", "uint32", True),
                KernelArgument("timeout_ticks", "uint64", False),
            )
        ),
    )

    context.launch(
        kernel,
        released_buffer,
        timeout_buffer,
        np.uint64(100_000),
        config=LaunchConfig(block_dim=(2, 2, 2)),
    )
    context.copy_to_host(result, timeout_buffer)

    assert result[0] == timed_out


def test_clock_counters_recorded_on_processor_id_zero(
    context: DeviceContext, profiling_module: DeviceModule
) -> None:
    driver = ctypes.CDLL("libcuda.so.1")
    device = ctypes.c_int()
    processor_count = ctypes.c_int()
    assert driver.cuDeviceGet(ctypes.byref(device), context.device_id) == 0
    assert driver.cuDeviceGetAttribute(ctypes.byref(processor_count), 16, device) == 0

    steady = np.zeros(1, dtype=np.uint64)
    cycles = np.zeros(1, dtype=np.uint64)
    recorded = np.zeros(1, dtype=np.uint32)
    arrays = (steady, cycles, recorded)
    buffers = tuple(context.create_buffer(1, dtype=array.dtype) for array in arrays)
    for buffer, array in zip(buffers, arrays, strict=True):
        context.copy_to_device(buffer, array)

    kernel = profiling_module.get_kernel(
        "recordClockCounters",
        spec=KernelSpec(
            (
                KernelArgument("steady_counter", "uint64", True),
                KernelArgument("cycle_counter", "uint64", True),
                KernelArgument("recorded", "uint32", True),
            )
        ),
    )

    context.launch(
        kernel, *buffers, config=LaunchConfig(grid_dim=(processor_count.value, 1, 1))
    )
    for buffer, array in zip(buffers, arrays, strict=True):
        context.copy_to_host(array, buffer)

    assert recorded[0] == 1
    assert steady[0] > 0
    assert cycles[0] > 0
