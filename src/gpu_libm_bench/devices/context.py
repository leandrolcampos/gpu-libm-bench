# Copyright 2026 Leandro A. Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""GPU device information and execution contexts."""

from dataclasses import dataclass
from types import ModuleType

from gpu_libm_bench.compiler.backends import get_backend

_SUPPORTED_BACKENDS = ("cuda",)


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    backend: str
    device_id: int
    name: str
    architecture: str
    total_memory_bytes: int


class DeviceContext:
    @staticmethod
    def supported_backends() -> tuple[str, ...]:
        return _SUPPORTED_BACKENDS

    @staticmethod
    def device_count(*, backend: str = "cuda") -> int:
        return _load_backend(backend).device_count()

    @staticmethod
    def get_device_info(device_id: int = 0, *, backend: str = "cuda") -> DeviceInfo:
        properties = _load_backend(backend).get_device_info(device_id)
        return DeviceInfo(backend=backend, **properties)


def _load_backend(backend: str) -> ModuleType:
    if backend not in _SUPPORTED_BACKENDS:
        raise ValueError(f"Unsupported backend: {backend!r}.")
    try:
        return get_backend(backend)
    except (ImportError, OSError) as error:
        raise RuntimeError(
            f"The {backend.upper()} backend could not be loaded: {error}"
        ) from error
