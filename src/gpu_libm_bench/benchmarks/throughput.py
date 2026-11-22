# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Independent-call throughput launch planning, measurement, and analysis."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import chain
from math import inf, isfinite, prod
from operator import index
from time import monotonic, sleep
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from gpu_libm_bench.devices import (
    DeviceBuffer,
    DeviceContext,
    DeviceInfo,
    DeviceKernel,
    KernelInfo,
    KernelMetric,
    LaunchConfig,
)
from gpu_libm_bench.inputs import InputBatch
from gpu_libm_bench.kernels import KernelKind, KernelRequest
from gpu_libm_bench.types import Scalar

from ._throttle import recovery_delay_s

_WARP_CANDIDATES = (8, 4)
_WAVE_CANDIDATES = (4, 2, 1)


@dataclass(frozen=True, slots=True)
class ThroughputPlannedLaunch:
    request: KernelRequest
    kernel_info: KernelInfo
    config: LaunchConfig
    blocks_per_multiprocessor: int
    batch_size: int


@dataclass(frozen=True, slots=True)
class ThroughputExcludedLaunch:
    request: KernelRequest
    reason: str


def try_plan_throughput_launch(
    device_info: DeviceInfo,
    kernel: DeviceKernel,
    request: KernelRequest,
    *,
    batch_size: int,
) -> ThroughputPlannedLaunch | ThroughputExcludedLaunch:
    unroll_factor = request.unroll_factor
    if request.kind is not KernelKind.THROUGHPUT or unroll_factor is None:
        raise ValueError(
            "expected a throughput kernel request with an unroll factor, "
            f"got kind={request.kind!r}, unroll_factor={unroll_factor!r}"
        )
    batch_size = _as_integer(batch_size, "batch_size", minimum=1)
    max_batch_size = np.iinfo(np.uint64).max
    if batch_size > max_batch_size:
        raise ValueError(
            f"batch_size={batch_size} input tuples per launch exceeds "
            f"the maximum representable uint64 value ({max_batch_size})"
        )

    limits = device_info.launch_limits
    warp_size = device_info.warp_size
    try:
        kernel_info = kernel.info
        max_threads = min(
            _WARP_CANDIDATES[0] * warp_size,
            limits.max_block_dim[0],
            limits.max_threads_per_block,
            kernel_info.max_threads_per_block,
        )
        candidates = _block_candidates(kernel, warp_size, max_threads // warp_size)
    except (RuntimeError, ValueError) as error:
        raise type(error)(
            f"{request.library} {request.symbol}: launch planning failed: {error}"
        ) from error

    if not candidates:
        return ThroughputExcludedLaunch(request, "no viable whole-warp block size")

    for warps, blocks_per_multiprocessor in candidates:
        block_size = warps * warp_size
        resident_grid = device_info.multiprocessor_count * blocks_per_multiprocessor
        max_grid = min(
            (batch_size + block_size - 1) // block_size, limits.max_grid_dim[0]
        )
        for waves in _WAVE_CANDIDATES:
            grid_size = min(waves * resident_grid, max_grid)
            if (
                unroll_factor > 1
                and batch_size < unroll_factor * grid_size * block_size
            ):
                continue
            return ThroughputPlannedLaunch(
                request=request,
                kernel_info=kernel_info,
                config=LaunchConfig(
                    grid_dim=(grid_size, 1, 1), block_dim=(block_size, 1, 1)
                ),
                blocks_per_multiprocessor=blocks_per_multiprocessor,
                batch_size=batch_size,
            )
    return ThroughputExcludedLaunch(
        request,
        f"batch_size={batch_size} cannot give each thread {unroll_factor} evaluations "
        "at any candidate launch geometry",
    )


def _block_candidates(
    kernel: DeviceKernel, warp_size: int, max_warps: int
) -> list[tuple[int, int]]:
    if max_warps == 0:
        return []

    primary = dict.fromkeys(min(warps, max_warps) for warps in _WARP_CANDIDATES)
    candidates = []
    for warps in primary:
        blocks_per_multiprocessor = kernel.max_active_blocks_per_multiprocessor(
            threads_per_block=warps * warp_size
        )
        if blocks_per_multiprocessor > 0:
            candidates.append((warps, blocks_per_multiprocessor))
    if candidates:
        return sorted(
            candidates, key=lambda pair: (pair[0] * pair[1], pair[0]), reverse=True
        )

    for warps in range(max_warps, 0, -1):
        if warps not in primary:
            blocks_per_multiprocessor = kernel.max_active_blocks_per_multiprocessor(
                threads_per_block=warps * warp_size
            )
            if blocks_per_multiprocessor > 0:
                return [(warps, blocks_per_multiprocessor)]
    return []


@dataclass(frozen=True, slots=True)
class ThroughputAttempt:
    batch_index: int
    duration_ms: float
    clock_frequency_hz: float | None
    discard_reason: str | None


@dataclass(frozen=True, slots=True)
class ThroughputMeasurement:
    domain: str
    device_info: DeviceInfo
    launch: ThroughputPlannedLaunch
    sample_size: int
    warmup_runs: int
    min_clock_hz: float
    attempts: tuple[ThroughputAttempt, ...]
    timed_out: bool

    @property
    def accepted_count(self) -> int:
        return sum(attempt.discard_reason is None for attempt in self.attempts)

    @property
    def complete(self) -> bool:
        return not self.timed_out and self.accepted_count == self.sample_size


@dataclass(frozen=True, slots=True)
class ThroughputResult:
    measurement: ThroughputMeasurement
    mean: float
    stdev: float | None
    min: float
    max: float
    unit: ClassVar[str] = "Gcall/s"


class ThroughputBenchmark:
    def __init__(
        self,
        context: DeviceContext,
        kernel: DeviceKernel,
        launch: ThroughputPlannedLaunch,
        *,
        sample_size: int,
        timeout_s: float | None = None,
        warmup_runs: int,
        min_clock_ratio: float = 0.0,
        initial_delay_s: float = 0.0,
    ) -> None:
        arguments = kernel.spec.arguments
        input_arguments = arguments[:-2]
        if not input_arguments or not all(
            argument.is_pointer for argument in input_arguments
        ):
            raise ValueError(
                "throughput kernel must declare at least one input pointer "
                "before checksums and count"
            )
        checksums, count = arguments[-2:]
        if (
            checksums.name != "checksums"
            or not checksums.is_pointer
            or checksums.dtype not in {"uint16", "uint32", "uint64"}
        ):
            raise ValueError(
                "the penultimate throughput kernel argument must be "
                "a uint16, uint32, or uint64 pointer named checksums"
            )
        if count.name != "count" or count.is_pointer or count.dtype != "uint64":
            raise ValueError(
                "the last throughput argument must be a uint64 scalar named count"
            )

        sample_size = _as_integer(sample_size, "sample_size", minimum=1)
        warmup_runs = _as_integer(warmup_runs, "warmup_runs", minimum=0)
        if timeout_s is not None and (not isfinite(timeout_s) or timeout_s <= 0):
            raise ValueError("timeout_s must be finite and positive")
        if not isfinite(min_clock_ratio) or min_clock_ratio < 0:
            raise ValueError("min_clock_ratio must be finite and nonnegative")
        if not isfinite(initial_delay_s) or initial_delay_s < 0:
            raise ValueError("initial_delay_s must be finite and nonnegative")

        self._context = context
        self._kernel = kernel
        self._launch = launch
        self._input_arguments = input_arguments
        self._checksum_dtype = checksums.dtype
        self._sample_size = sample_size
        self._timeout_s = timeout_s
        self._warmup_runs = warmup_runs
        self._min_clock_ratio = min_clock_ratio
        self._initial_delay_s = initial_delay_s

    def measure(
        self, domain: str, batches: Iterable[InputBatch]
    ) -> ThroughputMeasurement:
        context = self._context
        launch = self._launch
        device_info = context.device_info
        clock_threshold = self._min_clock_ratio * device_info.default_clock_frequency_hz
        if self._min_clock_ratio > 0 and (
            not isfinite(clock_threshold) or clock_threshold <= 0
        ):
            raise ValueError(
                "min_clock_ratio times the device default clock frequency "
                f"must be finite and positive in Hz, got {clock_threshold}"
            )

        input_buffers = tuple(
            context.create_buffer(launch.batch_size, dtype=argument.dtype)
            for argument in self._input_arguments
        )
        checksums = context.create_buffer(
            prod(launch.config.grid_dim) * prod(launch.config.block_dim),
            dtype=self._checksum_dtype,
        )
        arguments: tuple[DeviceBuffer[Any] | Scalar, ...] = (
            *input_buffers,
            checksums,
            np.uint64(launch.batch_size),
        )
        batches = iter(batches)
        if self._warmup_runs:
            first = _next_batch(batches, domain, 0)
            self._copy_inputs(input_buffers, self._batch_arrays(first, domain))
            for _ in range(self._warmup_runs):
                context.launch(self._kernel, *arguments, config=launch.config)
            batches = chain((first,), batches)

        deadline = inf if self._timeout_s is None else monotonic() + self._timeout_s
        attempts: list[ThroughputAttempt] = []
        timed_out = self._collect(
            domain,
            batches,
            input_buffers,
            arguments,
            clock_threshold,
            deadline,
            attempts,
        )
        return ThroughputMeasurement(
            domain=domain,
            device_info=device_info,
            launch=launch,
            sample_size=self._sample_size,
            warmup_runs=self._warmup_runs,
            min_clock_hz=clock_threshold,
            attempts=tuple(attempts),
            timed_out=timed_out,
        )

    def _collect(
        self,
        domain: str,
        batches: Iterator[InputBatch],
        input_buffers: tuple[DeviceBuffer[Scalar], ...],
        arguments: tuple[DeviceBuffer[Any] | Scalar, ...],
        clock_threshold: float,
        deadline: float,
        attempts: list[ThroughputAttempt],
    ) -> bool:
        metrics = {KernelMetric.DURATION}
        if clock_threshold > 0:
            metrics.add(KernelMetric.CLOCK_FREQUENCY)

        delay_s = self._initial_delay_s
        for batch_index in range(self._sample_size):
            if monotonic() >= deadline:
                return True
            arrays = self._batch_arrays(
                _next_batch(batches, domain, batch_index), domain
            )
            consecutive_discards = 0
            while monotonic() < deadline:
                self._copy_inputs(input_buffers, arrays)
                if monotonic() >= deadline:
                    return True
                profile = self._context.launch_and_profile(
                    self._kernel,
                    *arguments,
                    config=self._launch.config,
                    metrics=metrics,
                )
                remaining_s = deadline - monotonic()
                duration_ms = profile.duration_ms
                if duration_ms is None or not isfinite(duration_ms) or duration_ms <= 0:
                    request = self._launch.request
                    raise RuntimeError(
                        f"{request.library} {request.symbol} ({domain}): "
                        f"GPU duration for batch index {batch_index} must be finite "
                        f"and positive in ms, got {duration_ms!r}"
                    )

                clock_hz = profile.clock_frequency_hz
                reason = _discard_reason(clock_hz, clock_threshold, remaining_s)
                attempts.append(
                    ThroughputAttempt(batch_index, duration_ms, clock_hz, reason)
                )
                if remaining_s <= 0:
                    return True
                if reason is None:
                    break

                consecutive_discards += 1
                delay_s = recovery_delay_s(delay_s, consecutive_discards)
                if delay_s > 0:
                    sleep(min(delay_s, remaining_s))
            else:
                return True
        return False

    def _batch_arrays(
        self, batch: InputBatch, domain: str
    ) -> tuple[NDArray[Scalar], ...]:
        if batch.domain != domain:
            raise ValueError(
                f"expected batch for domain {domain!r}, got {batch.domain!r}"
            )
        expected_names = {argument.name for argument in self._input_arguments}
        if batch.arguments.keys() != expected_names:
            raise ValueError(
                f"expected input arguments {sorted(expected_names)} for domain "
                f"{domain!r}, got {sorted(batch.arguments)}"
            )

        arrays = []
        for argument in self._input_arguments:
            source = batch.arguments[argument.name]
            label = f"throughput input {argument.name!r}"
            if not isinstance(source, np.ndarray):
                raise TypeError(
                    f"{label} must be a NumPy array, got {type(source).__name__}"
                )
            if source.dtype != np.dtype(argument.dtype):
                raise ValueError(
                    f"{label} must have dtype {argument.dtype}, got {source.dtype}"
                )
            if source.shape != (self._launch.batch_size,):
                raise ValueError(
                    f"{label} must have shape ({self._launch.batch_size},) "
                    f"to match batch_size, got {source.shape}"
                )
            arrays.append(np.ascontiguousarray(source))
        return tuple(arrays)

    def _copy_inputs(
        self,
        buffers: tuple[DeviceBuffer[Scalar], ...],
        arrays: tuple[NDArray[Scalar], ...],
    ) -> None:
        for buffer, array in zip(buffers, arrays, strict=True):
            self._context.copy_to_device(buffer, array)


def analyze(measurement: ThroughputMeasurement) -> ThroughputResult | None:
    if not measurement.complete:
        return None

    durations = np.array(
        [
            attempt.duration_ms
            for attempt in measurement.attempts
            if attempt.discard_reason is None
        ],
        dtype=np.float64,
    )
    gigacalls_per_batch = measurement.launch.batch_size / 1e9
    durations_s = durations * 1e-3
    rates = gigacalls_per_batch / durations_s
    return ThroughputResult(
        measurement=measurement,
        mean=float(gigacalls_per_batch / durations_s.mean()),
        stdev=float(rates.std(ddof=1)) if rates.size > 1 else None,
        min=float(rates.min()),
        max=float(rates.max()),
    )


def _as_integer(value: int, name: str, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    try:
        value = index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer") from None
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _next_batch(
    batches: Iterator[InputBatch], domain: str, batch_index: int
) -> InputBatch:
    batch = next(batches, None)
    if batch is None:
        raise ValueError(
            f"throughput batches for domain {domain!r} ended before "
            f"batch index {batch_index}"
        )
    return batch


def _discard_reason(
    clock_hz: float | None, clock_threshold: float, remaining_s: float
) -> str | None:
    if remaining_s <= 0:
        return "timeout"
    if clock_threshold > 0:
        if clock_hz is None or not isfinite(clock_hz) or clock_hz <= 0:
            return "invalid clock"
        if clock_hz < clock_threshold:
            return "clock below threshold"
    return None
