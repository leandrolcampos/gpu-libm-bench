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

"""GPU device images and modules."""

from dataclasses import dataclass
from operator import index
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from gpu_libm_bench.kernels import KernelSpec
from gpu_libm_bench.types import Scalar

from .memory import DeviceBuffer

if TYPE_CHECKING:
    from gpu_libm_bench.backends import _cuda  # pyright: ignore[reportMissingModuleSource]
    from gpu_libm_bench.compiler.targets import CompilationTarget


@dataclass(frozen=True, slots=True)
class DeviceImage:
    image_path: Path
    target: "CompilationTarget"
    library: str
    functions: tuple[str, ...] | None
    manifest_path: Path | None


@dataclass(frozen=True, slots=True)
class KernelInfo:
    max_threads_per_block: int
    registers_per_thread: int
    static_shared_memory_bytes: int
    local_memory_bytes_per_thread: int


class DeviceKernel:
    def __init__(self, native: "_cuda.DeviceKernel", spec: KernelSpec) -> None:
        self._native = native
        self._spec = spec
        self._info: KernelInfo | None = None
        self._occupancy: dict[tuple[int, int], int] = {}

    @property
    def spec(self) -> KernelSpec:
        return self._spec

    @property
    def info(self) -> KernelInfo:
        if self._info is None:
            self._info = KernelInfo(**self._native.info)
        return self._info

    def max_active_blocks_per_multiprocessor(
        self, *, threads_per_block: int, dynamic_shared_memory_bytes: int = 0
    ) -> int:
        threads_per_block = _normalize_integer(threads_per_block, "threads_per_block")
        dynamic_shared_memory_bytes = _normalize_integer(
            dynamic_shared_memory_bytes, "dynamic_shared_memory_bytes"
        )
        if threads_per_block <= 0:
            raise ValueError("threads_per_block must be positive")
        if dynamic_shared_memory_bytes < 0:
            raise ValueError("dynamic_shared_memory_bytes must be nonnegative")

        configuration = (threads_per_block, dynamic_shared_memory_bytes)
        blocks_per_multiprocessor = self._occupancy.get(configuration)
        if blocks_per_multiprocessor is None:
            blocks_per_multiprocessor = (
                self._native.max_active_blocks_per_multiprocessor(
                    threads_per_block=threads_per_block,
                    dynamic_shared_memory_bytes=dynamic_shared_memory_bytes,
                )
            )
            self._occupancy[configuration] = blocks_per_multiprocessor
        return blocks_per_multiprocessor

    def _validate_arguments(
        self, arguments: tuple[DeviceBuffer[Any] | Scalar, ...]
    ) -> None:
        if len(arguments) != len(self.spec.arguments):
            raise ValueError("kernel argument count does not match its specification")

        for parameter, argument in zip(self.spec.arguments, arguments, strict=True):
            label = f"kernel argument {parameter.name!r}"

            if parameter.is_pointer:
                if not isinstance(argument, DeviceBuffer):
                    raise TypeError(f"{label} expects a device buffer")
            elif not isinstance(argument, np.generic):
                raise TypeError(f"{label} expects a NumPy scalar")

            if argument.dtype != np.dtype(parameter.dtype):
                raise ValueError(f"{label} has the wrong dtype")


class DeviceModule:
    def __init__(self, native: "_cuda.DeviceModule", image: DeviceImage) -> None:
        self._native = native
        self._image = image

    @property
    def image(self) -> DeviceImage:
        return self._image

    def get_kernel(self, symbol: str, *, spec: KernelSpec) -> DeviceKernel:
        return DeviceKernel(self._native.get_kernel(symbol), spec)


def _normalize_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    try:
        return index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer") from None
