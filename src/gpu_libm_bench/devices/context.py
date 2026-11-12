# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""GPU device information and execution contexts."""

from dataclasses import dataclass
from operator import index
from types import ModuleType, TracebackType
from typing import TYPE_CHECKING, Any, overload

import numpy as np
from numpy.typing import DTypeLike, NDArray

from gpu_libm_bench.compiler.backends import get_backend
from gpu_libm_bench.types import Scalar

from .memory import DeviceBuffer, _buffer_dtype
from .module import DeviceImage, DeviceKernel, DeviceModule

if TYPE_CHECKING:
    from gpu_libm_bench.backends import _cuda  # pyright: ignore[reportMissingModuleSource]

_SUPPORTED_BACKENDS = ("cuda",)


@dataclass(frozen=True, slots=True)
class TimerInfo:
    source: str
    counter_bits: int
    frequency_hz: int

    def __post_init__(self) -> None:
        if self.frequency_hz <= 0:
            raise ValueError("timer frequency must be positive")

    def to_nanoseconds(self, ticks: NDArray[np.uint64]) -> NDArray[np.float64]:
        return ticks.astype(np.float64) * (1_000_000_000 / self.frequency_hz)


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    backend: str
    id: int
    name: str
    architecture: str
    total_memory_bytes: int
    timer_info: TimerInfo


@dataclass(frozen=True, slots=True)
class LaunchConfig:
    grid: tuple[int, int, int] = (1, 1, 1)
    block: tuple[int, int, int] = (1, 1, 1)
    shared_memory_bytes: int = 0

    def __post_init__(self) -> None:
        for name, dimensions in (("grid", self.grid), ("block", self.block)):
            try:
                normalized = tuple(index(dimension) for dimension in dimensions)
            except TypeError as error:
                raise TypeError(f"{name} dimensions must be integers") from error

            if len(normalized) != 3:
                raise ValueError(f"{name} must have exactly three dimensions")

            if any(dimension <= 0 for dimension in normalized):
                raise ValueError(f"{name} dimensions must be positive")

            object.__setattr__(self, name, normalized)

        try:
            shared_memory_bytes = index(self.shared_memory_bytes)
        except TypeError as error:
            raise TypeError("shared_memory_bytes must be an integer") from error

        if shared_memory_bytes < 0:
            raise ValueError("shared_memory_bytes must be nonnegative")

        object.__setattr__(self, "shared_memory_bytes", shared_memory_bytes)


class DeviceContext:
    def __init__(self, device_id: int = 0, *, backend: str = "cuda") -> None:
        self._native: _cuda.DeviceContext = _load_backend(backend).DeviceContext(
            device_id
        )

        self._backend = backend
        self._device_id = device_id
        self._device_info: DeviceInfo | None = None

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def device_id(self) -> int:
        return self._device_id

    @property
    def device_info(self) -> DeviceInfo:
        if self._device_info is None:
            self._device_info = self.get_device_info(
                self.device_id, backend=self.backend
            )
        return self._device_info

    @property
    def released(self) -> bool:
        return self._native.released

    def release(self) -> None:
        self._native.release()

    def __enter__(self) -> "DeviceContext":
        self._ensure_active()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            self.release()
        except RuntimeError as error:
            if exc_value is None:
                raise

            exc_value.add_note(f"device context cleanup failed: {error}")

    def load_module(self, image: DeviceImage) -> DeviceModule:
        self._ensure_active()

        if image.target.backend != self.backend:
            raise ValueError("device image backend does not match the device context")
        if not image.image_path.is_file():
            raise FileNotFoundError(f"device image was not found: {image.image_path}")

        return DeviceModule(self._native.load_module(str(image.image_path)), image)

    @overload
    def create_buffer[T: Scalar](
        self, size: int, *, dtype: type[T] | np.dtype[T]
    ) -> DeviceBuffer[T]: ...

    @overload
    def create_buffer(self, size: int, *, dtype: DTypeLike) -> DeviceBuffer[Any]: ...

    def create_buffer(self, size: int, *, dtype: DTypeLike) -> DeviceBuffer[Any]:
        self._ensure_active()

        size = index(size)
        if size < 0:
            raise ValueError("device buffer size must be nonnegative")

        resolved = _buffer_dtype(dtype)

        return DeviceBuffer(
            self._native.create_buffer(size * resolved.itemsize), resolved
        )

    def copy_to_device[T: Scalar](
        self, destination: DeviceBuffer[T], source: NDArray[T]
    ) -> None:
        self._ensure_active()

        if not isinstance(destination, DeviceBuffer):
            raise TypeError("destination must be a DeviceBuffer")
        if not isinstance(source, np.ndarray):
            raise TypeError("source must be a NumPy array")

        if source.ndim != 1 or not source.flags.c_contiguous:
            raise ValueError("source must be a contiguous one-dimensional array")

        if destination.dtype != source.dtype:
            raise ValueError("source and destination must have the same dtype")
        if destination.size < source.size:
            raise ValueError("copy exceeds destination capacity")

        self._native.copy_to_device(destination._native, source)

    def copy_to_host[T: Scalar](
        self, destination: NDArray[T], source: DeviceBuffer[T]
    ) -> None:
        self._ensure_active()

        if not isinstance(destination, np.ndarray):
            raise TypeError("destination must be a NumPy array")
        if not isinstance(source, DeviceBuffer):
            raise TypeError("source must be a DeviceBuffer")

        if destination.ndim != 1 or not destination.flags.c_contiguous:
            raise ValueError("destination must be a contiguous one-dimensional array")
        if not destination.flags.writeable:
            raise ValueError("destination must be writable")

        if destination.dtype != source.dtype:
            raise ValueError("source and destination must have the same dtype")
        if destination.size < source.size:
            raise ValueError("copy exceeds destination capacity")

        self._native.copy_to_host(destination, source._native)

    def launch(
        self,
        kernel: DeviceKernel,
        *arguments: DeviceBuffer[Any] | Scalar,
        config: LaunchConfig,
    ) -> None:
        self._ensure_active()
        kernel._validate_arguments(arguments)

        native_arguments = tuple(
            argument._native if isinstance(argument, DeviceBuffer) else argument
            for argument in arguments
        )

        self._native.launch(
            kernel._native,
            native_arguments,
            config.grid,
            config.block,
            config.shared_memory_bytes,
        )

    def _ensure_active(self) -> None:
        if self.released:
            raise RuntimeError("device context has been released")

    @staticmethod
    def supported_backends() -> tuple[str, ...]:
        return _SUPPORTED_BACKENDS

    @staticmethod
    def device_count(*, backend: str = "cuda") -> int:
        return _load_backend(backend).device_count()

    @staticmethod
    def get_device_info(device_id: int = 0, *, backend: str = "cuda") -> DeviceInfo:
        properties = _load_backend(backend).get_device_info(device_id)
        return DeviceInfo(
            backend=backend,
            id=properties["id"],
            name=properties["name"],
            architecture=properties["architecture"],
            total_memory_bytes=properties["total_memory_bytes"],
            timer_info=TimerInfo(**properties["timer_info"]),
        )


def _load_backend(backend: str) -> ModuleType:
    if backend not in _SUPPORTED_BACKENDS:
        raise ValueError(f"unsupported backend: {backend!r}")

    try:
        return get_backend(backend)
    except (ImportError, OSError) as error:
        raise RuntimeError(
            f"the {backend.upper()} backend could not be loaded: {error}"
        ) from error
