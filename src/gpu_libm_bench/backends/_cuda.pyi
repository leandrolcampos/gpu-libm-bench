# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Python interface to the CUDA backend."""

from collections.abc import Set
from enum import Enum
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

class LaunchLimits(TypedDict):
    max_grid_dim: tuple[int, int, int]
    max_block_dim: tuple[int, int, int]
    max_threads_per_block: int
    max_shared_memory_bytes_per_block: int

class TimerInfo(TypedDict):
    source: str
    counter_bits: int
    frequency_hz: int

class DeviceInfo(TypedDict):
    id: int
    name: str
    architecture: str
    multiprocessor_count: int
    warp_size: int
    default_clock_frequency_hz: int
    total_memory_bytes: int
    l2_cache_size_bytes: int
    launch_limits: LaunchLimits
    timer_info: TimerInfo

class KernelMetric(Enum):
    CLOCK_FREQUENCY = ...
    DURATION = ...

class KernelProfile(TypedDict):
    clock_frequency_hz: float | None
    duration_ms: float | None

class DeviceBuffer:
    @property
    def nbytes(self) -> int: ...

class KernelInfo(TypedDict):
    max_threads_per_block: int
    registers_per_thread: int
    static_shared_memory_bytes: int
    local_memory_bytes_per_thread: int

class DeviceKernel:
    @property
    def info(self) -> KernelInfo: ...
    def max_active_blocks_per_multiprocessor(
        self, *, threads_per_block: int, dynamic_shared_memory_bytes: int = 0
    ) -> int: ...

class DeviceModule:
    def get_kernel(self, symbol: str) -> DeviceKernel: ...

class DeviceContext:
    def __init__(self, device_id: int = 0) -> None: ...
    def release(self) -> None: ...
    @property
    def released(self) -> bool: ...
    def create_buffer(self, byte_count: int) -> DeviceBuffer: ...
    def copy_to_device(
        self, destination: DeviceBuffer, source: NDArray[np.generic]
    ) -> None: ...
    def copy_to_host(
        self, destination: NDArray[np.generic], source: DeviceBuffer
    ) -> None: ...
    def load_module(self, path: str) -> DeviceModule: ...
    def launch(
        self,
        kernel: DeviceKernel,
        kernel_arguments: tuple[DeviceBuffer | np.generic, ...],
        grid_dim: tuple[int, int, int],
        block_dim: tuple[int, int, int],
        shared_memory_bytes: int,
    ) -> None: ...
    def profile(
        self,
        kernel: DeviceKernel,
        kernel_arguments: tuple[DeviceBuffer | np.generic, ...],
        grid_dim: tuple[int, int, int],
        block_dim: tuple[int, int, int],
        shared_memory_bytes: int,
        metrics: Set[KernelMetric],
    ) -> KernelProfile: ...

def device_count() -> int: ...
def get_device_info(device_id: int = 0) -> DeviceInfo: ...
def _set_images_build_root(root: str) -> None: ...
