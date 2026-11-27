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

"""Integration tests for throughput measurements of scalar signatures."""

import os
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from gpu_libm_bench.compiler.images import resolve_device_image
from gpu_libm_bench.compiler.targets import CompilationTarget
from gpu_libm_bench.devices import DeviceContext, DeviceImage, LaunchConfig
from gpu_libm_bench.kernels import KernelArgument, KernelSpec

pytestmark = [pytest.mark.cuda, pytest.mark.execution]


@pytest.mark.parametrize(
    ("symbol", "sources"),
    [
        pytest.param(
            "halfBinaryThroughputProbe",
            (
                np.array([0x3E00, 0x8001, 0x7C00], dtype=np.uint16).view(np.float16),
                np.array([0x8000, 0x0003, 0x7E35], dtype=np.uint16).view(np.float16),
            ),
            id="half-binary",
        ),
        pytest.param(
            "mixedThroughputProbe",
            (
                np.array([-0x123456789, 0x100000001, -0x300000007], dtype=np.int64),
                np.array([0xFFFFFFFF, 0x80000001, 17], dtype=np.uint32),
                np.array([-7, 3, -19], dtype=np.int32),
                np.array(
                    [0xBFF4000000000001, 0x0000000100000003, 0x7FF8123456789ABC],
                    dtype=np.uint64,
                ).view(np.float64),
            ),
            id="mixed-integer-and-double",
        ),
    ],
)
def test_throughput_preserves_all_arguments_for_scalar_signatures(
    symbol: str, sources: tuple[NDArray[np.generic], ...]
) -> None:
    pass_count, count = 2, 3
    source_bits = tuple(source.view(f"uint{source.itemsize * 8}") for source in sources)
    # Both probes return their last argument.
    return_bits = source_bits[-1]
    checksum = np.zeros(1, dtype=return_bits.dtype)
    trace = np.zeros(count * pass_count * len(sources), dtype=np.uint64)
    calls = np.zeros(1, dtype=np.uint64)

    spec = KernelSpec(
        (
            KernelArgument("pass_count", "uint32", False),
            KernelArgument("mask", "uint32", False),
            KernelArgument("count", "uint64", False),
            KernelArgument("checksums", checksum.dtype.name, True),
            KernelArgument("observed_inputs", "uint64", True),
            KernelArgument("observed_count", "uint64", True),
            *(
                KernelArgument(f"input_{index}", source.dtype.name, True)
                for index, source in enumerate(sources)
            ),
        )
    )

    with DeviceContext() as context:
        target = CompilationTarget(
            "cuda", "nvptx64-nvidia-cuda", context.device_info.architecture
        )
        configured = os.environ.get("GPU_LIBM_BENCH_DEVICE_IMAGE")
        image = resolve_device_image(
            "llvm-libm",
            target=target,
            image_path=Path(configured) if configured is not None else None,
        )
        path = image.image_path.parent / "tests" / "throughput.cubin"
        module = context.load_module(DeviceImage(path, target, "tests", (), None))
        kernel = module.get_kernel(symbol, spec=spec)

        outputs = (checksum, trace, calls)
        arrays = outputs + sources
        buffers = tuple(
            context.create_buffer(array.size, dtype=array.dtype) for array in arrays
        )
        for buffer, array in zip(buffers, arrays, strict=True):
            context.copy_to_device(buffer, array)

        context.launch(
            kernel,
            np.uint32(pass_count),
            np.uint32(0),
            np.uint64(count),
            *buffers,
            config=LaunchConfig(),
        )
        for array, buffer in zip(outputs, buffers[: len(outputs)], strict=True):
            context.copy_to_host(array, buffer)

    # The first two inputs run together for both passes, then the tail runs twice.
    expected_inputs = np.column_stack(source_bits)[[0, 1, 0, 1, 2, 2]]
    assert calls[0] == count * pass_count
    np.testing.assert_array_equal(trace.reshape(expected_inputs.shape), expected_inputs)
    assert checksum[0] == np.bitwise_xor.reduce(return_bits + return_bits)
