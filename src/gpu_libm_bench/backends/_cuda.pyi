# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Python interface to the CUDA backend."""

from collections.abc import Set
from enum import Enum
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

class KernelMetric(Enum):
    AVERAGE_FREQUENCY = ...
    DURATION = ...

class KernelProfile(TypedDict):
    average_frequency_hz: float | None
    duration_ms: float | None

class TimerInfo(TypedDict):
    source: str
    counter_bits: int
    frequency_hz: int

class DeviceInfo(TypedDict):
    id: int
    name: str
    architecture: str
    total_memory_bytes: int
    timer_info: TimerInfo

def device_count() -> int: ...
def get_device_info(device_id: int = 0) -> DeviceInfo: ...

class DeviceBuffer:
    @property
    def nbytes(self) -> int: ...

class DeviceKernel: ...

class DeviceModule:
    def get_kernel(self, symbol: str) -> DeviceKernel: ...

class DeviceContext:
    def __init__(self, device_id: int = 0) -> None: ...
    @property
    def released(self) -> bool: ...
    def release(self) -> None: ...
    def create_buffer(self, byte_count: int) -> DeviceBuffer: ...
    def load_module(self, path: str) -> DeviceModule: ...
    def copy_to_device(
        self, destination: DeviceBuffer, source: NDArray[np.generic]
    ) -> None: ...
    def copy_to_host(
        self, destination: NDArray[np.generic], source: DeviceBuffer
    ) -> None: ...
    def launch(
        self,
        kernel: DeviceKernel,
        kernel_arguments: tuple[DeviceBuffer | np.generic, ...],
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        shared_memory_bytes: int,
    ) -> None: ...
    def profile(
        self,
        kernel: DeviceKernel,
        kernel_arguments: tuple[DeviceBuffer | np.generic, ...],
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        shared_memory_bytes: int,
        metrics: Set[KernelMetric],
    ) -> KernelProfile: ...
