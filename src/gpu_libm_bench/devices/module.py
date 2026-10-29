# Copyright 2026 Leandro A. Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""GPU device images and modules."""

from dataclasses import dataclass
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
    path: Path
    target: "CompilationTarget"
    library: str
    functions: tuple[str, ...] | None
    manifest: Path | None


class DeviceKernel:
    def __init__(self, native: "_cuda.DeviceKernel", spec: KernelSpec) -> None:
        self._native = native
        self._spec = spec

    @property
    def spec(self) -> KernelSpec:
        return self._spec

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
