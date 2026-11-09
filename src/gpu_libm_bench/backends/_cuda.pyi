# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Python interface to the CUDA backend."""

from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

class TimerProperties(TypedDict):
    source: str
    counter_bits: int
    frequency_hz: int

class DeviceProperties(TypedDict):
    device_id: int
    name: str
    architecture: str
    total_memory_bytes: int
    timer: TimerProperties

def device_count() -> int: ...
def get_device_info(device_id: int = 0) -> DeviceProperties: ...

class DeviceBuffer:
    @property
    def nbytes(self) -> int: ...

class DeviceKernel: ...

class DeviceModule:
    def get_kernel(self, symbol: str) -> DeviceKernel: ...

class DeviceContext:
    def __init__(self, device_id: int = 0) -> None: ...
    @property
    def closed(self) -> bool: ...
    def close(self) -> None: ...
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
        arguments: tuple[DeviceBuffer | np.generic, ...],
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        shared_memory_bytes: int,
    ) -> None: ...
