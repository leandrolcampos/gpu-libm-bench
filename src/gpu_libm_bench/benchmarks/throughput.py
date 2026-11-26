# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Independent-call throughput measurement, analysis, and launch planning."""

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import chain, zip_longest
from math import inf, isfinite, prod
from operator import index
from time import monotonic, sleep
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from gpu_libm_bench.configs import ThroughputSearchConfig
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
from gpu_libm_bench.kernels import KernelKind, KernelRequest, KernelSpec
from gpu_libm_bench.types import Scalar

from ._throttle import recovery_delay_s

_THROUGHPUT_PREFIX_ARGUMENTS = (
    ("pass_count", False, "uint32"),
    ("mask", False, "uint32"),
)
_THROUGHPUT_SUFFIX_ARGUMENTS = (
    ("checksums", True, None),
    ("count", False, "uint64"),
)


@dataclass(frozen=True, slots=True)
class ThroughputPlannedLaunch:
    request: KernelRequest
    kernel_info: KernelInfo
    config: LaunchConfig
    max_resident_blocks_per_multiprocessor: int
    batch_size: int

    @property
    def thread_count(self) -> int:
        return prod(self.config.grid_dim) * prod(self.config.block_dim)


# ===----------------------------------------------------------------------=== #
# Measurement
# ===----------------------------------------------------------------------=== #


@dataclass(frozen=True, slots=True)
class ThroughputAttempt:
    batch_index: int
    duration_ms: float
    clock_frequency_hz: float | None
    discard_reason: str | None


@dataclass(frozen=True, slots=True)
class ThroughputMeasurement:
    domain: str
    attempts: tuple[ThroughputAttempt, ...]
    device_info: DeviceInfo
    pass_count: int
    launch: ThroughputPlannedLaunch
    sample_size: int
    warmup_runs: int
    min_clock_frequency_hz: float
    timed_out: bool

    @property
    def evaluations_per_launch(self) -> int:
        return self.launch.batch_size * self.pass_count

    @property
    def accepted_count(self) -> int:
        return sum(attempt.discard_reason is None for attempt in self.attempts)

    @property
    def complete(self) -> bool:
        return not self.timed_out and self.accepted_count == self.sample_size


@dataclass(frozen=True, slots=True)
class ThroughputResult:
    measurement: ThroughputMeasurement
    aggregate: float
    stdev: float | None
    min: float
    max: float
    unit: ClassVar[str] = "Gcall/s"


@dataclass(slots=True)
class _ThroughputBuffers:
    inputs: tuple[DeviceBuffer[Scalar], ...]
    checksums: DeviceBuffer[Scalar]


class ThroughputBenchmark:
    def __init__(
        self,
        context: DeviceContext,
        kernel: DeviceKernel,
        launch: ThroughputPlannedLaunch,
        *,
        sample_size: int,
        timeout_s: float | None = None,
        pass_count: int = 1,
        warmup_runs: int,
        min_clock_ratio: float = 0.0,
        initial_delay_s: float = 0.0,
    ) -> None:
        arguments = kernel.spec.arguments
        prefix_length = len(_THROUGHPUT_PREFIX_ARGUMENTS)
        suffix_length = len(_THROUGHPUT_SUFFIX_ARGUMENTS)
        input_arguments = arguments[prefix_length:-suffix_length]
        if not input_arguments:
            raise ValueError(
                "throughput kernel must declare at least one input argument"
            )
        if not all(argument.is_pointer for argument in input_arguments):
            raise ValueError("throughput kernel input arguments must be pointers")

        throughput_dtypes = {}
        for argument, (name, is_pointer, dtype) in zip(
            arguments[:prefix_length] + arguments[-suffix_length:],
            _THROUGHPUT_PREFIX_ARGUMENTS + _THROUGHPUT_SUFFIX_ARGUMENTS,
            strict=True,
        ):
            if argument.name != name or argument.is_pointer != is_pointer:
                raise ValueError("kernel does not match the throughput protocol")
            actual_dtype = np.dtype(argument.dtype)
            if dtype is not None and actual_dtype != np.dtype(dtype):
                raise ValueError(f"{name} must have dtype {dtype}")
            throughput_dtypes[name] = actual_dtype

        checksum_dtype = throughput_dtypes["checksums"]
        if checksum_dtype.kind != "u" or checksum_dtype.itemsize not in (2, 4, 8):
            raise ValueError("checksums must have dtype uint16, uint32, or uint64")

        sample_size = _normalize_integer(sample_size, "sample_size", minimum=1)
        if timeout_s is not None and (not isfinite(timeout_s) or timeout_s <= 0):
            raise ValueError("timeout_s must be finite and positive")

        pass_count = _normalize_integer(pass_count, "pass_count", minimum=1)
        max_pass_count = np.iinfo(throughput_dtypes["pass_count"]).max
        if pass_count > max_pass_count:
            raise ValueError(f"pass_count must be at most {max_pass_count}")

        warmup_runs = _normalize_integer(warmup_runs, "warmup_runs", minimum=0)
        if not isfinite(min_clock_ratio) or min_clock_ratio < 0:
            raise ValueError("min_clock_ratio must be finite and nonnegative")
        if not isfinite(initial_delay_s) or initial_delay_s < 0:
            raise ValueError("initial_delay_s must be finite and nonnegative")

        self._context = context
        self._kernel = kernel
        self._launch = launch
        self._throughput_dtypes = throughput_dtypes
        self._input_arguments = input_arguments

        self._sample_size = sample_size
        self._timeout_s = timeout_s
        self._pass_count = throughput_dtypes["pass_count"].type(pass_count)
        self._warmup_runs = warmup_runs
        self._min_clock_ratio = min_clock_ratio
        self._initial_delay_s = initial_delay_s

    def measure(
        self, domain: str, batches: Iterable[InputBatch]
    ) -> ThroughputMeasurement:
        context = self._context
        launch = self._launch
        device_info = context.device_info
        min_clock_hz = self._min_clock_ratio * device_info.default_clock_frequency_hz
        if self._min_clock_ratio > 0 and (
            not isfinite(min_clock_hz) or min_clock_hz <= 0
        ):
            raise ValueError(
                "min_clock_ratio times the device default clock frequency "
                f"must be finite and positive in Hz, got {min_clock_hz}"
            )

        buffers = self._create_buffers()
        throughput_dtypes = self._throughput_dtypes
        arguments: tuple[DeviceBuffer[Any] | Scalar, ...] = (
            self._pass_count,
            throughput_dtypes["mask"].type(0),
            *buffers.inputs,
            buffers.checksums,
            throughput_dtypes["count"].type(launch.batch_size),
        )

        batch_iterator = iter(batches)
        if self._warmup_runs:
            first_batch = _get_next_batch(batch_iterator, domain, 0)
            self._copy_inputs_to_device(
                buffers.inputs, self._prepare_input_arrays(first_batch, domain)
            )
            for _ in range(self._warmup_runs):
                context.launch(self._kernel, *arguments, config=launch.config)
            batch_iterator = chain((first_batch,), batch_iterator)

        deadline = inf if self._timeout_s is None else monotonic() + self._timeout_s
        attempts: list[ThroughputAttempt] = []
        timed_out = self._collect_attempts(
            domain,
            batch_iterator,
            buffers.inputs,
            arguments,
            min_clock_hz,
            deadline,
            attempts,
        )

        return ThroughputMeasurement(
            domain=domain,
            attempts=tuple(attempts),
            device_info=device_info,
            pass_count=int(self._pass_count),
            launch=launch,
            sample_size=self._sample_size,
            warmup_runs=self._warmup_runs,
            min_clock_frequency_hz=min_clock_hz,
            timed_out=timed_out,
        )

    def _collect_attempts(
        self,
        domain: str,
        batches: Iterator[InputBatch],
        input_buffers: tuple[DeviceBuffer[Scalar], ...],
        arguments: tuple[DeviceBuffer[Any] | Scalar, ...],
        min_clock_hz: float,
        deadline: float,
        attempts: list[ThroughputAttempt],
    ) -> bool:
        metrics = {KernelMetric.DURATION}
        if min_clock_hz > 0:
            metrics.add(KernelMetric.CLOCK_FREQUENCY)

        delay_s = self._initial_delay_s
        for batch_index in range(self._sample_size):
            if monotonic() >= deadline:
                return True
            arrays = self._prepare_input_arrays(
                _get_next_batch(batches, domain, batch_index), domain
            )
            consecutive_discards = 0
            while monotonic() < deadline:
                self._copy_inputs_to_device(input_buffers, arrays)
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
                reason = _get_discard_reason(clock_hz, min_clock_hz, remaining_s)
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

    def _prepare_input_arrays(
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

    def _create_buffers(self) -> _ThroughputBuffers:
        batch_size = self._launch.batch_size
        return _ThroughputBuffers(
            inputs=tuple(
                self._context.create_buffer(batch_size, dtype=argument.dtype)
                for argument in self._input_arguments
            ),
            checksums=self._context.create_buffer(
                self._launch.thread_count, dtype=self._throughput_dtypes["checksums"]
            ),
        )

    def _copy_inputs_to_device(
        self,
        buffers: tuple[DeviceBuffer[Scalar], ...],
        arrays: tuple[NDArray[Scalar], ...],
    ) -> None:
        for destination, source in zip(buffers, arrays, strict=True):
            self._context.copy_to_device(destination, source)


def _get_next_batch(
    batches: Iterator[InputBatch], domain: str, batch_index: int
) -> InputBatch:
    batch = next(batches, None)
    if batch is None:
        raise ValueError(
            f"throughput batches for domain {domain!r} ended before "
            f"batch index {batch_index}"
        )
    return batch


def _get_discard_reason(
    clock_hz: float | None, min_clock_hz: float, remaining_s: float
) -> str | None:
    if remaining_s <= 0:
        return "timeout"
    if min_clock_hz > 0:
        if clock_hz is None or not isfinite(clock_hz) or clock_hz <= 0:
            return "invalid clock"
        if clock_hz < min_clock_hz:
            return "clock below threshold"
    return None


# ===----------------------------------------------------------------------=== #
# Analysis
# ===----------------------------------------------------------------------=== #


def analyze(measurement: ThroughputMeasurement) -> ThroughputResult | None:
    if not measurement.complete:
        return None

    durations_ms = np.array(
        [
            attempt.duration_ms
            for attempt in measurement.attempts
            if attempt.discard_reason is None
        ],
        dtype=np.float64,
    )

    gigacalls_per_batch = measurement.evaluations_per_launch / 1e9
    durations_s = durations_ms * 1e-3
    rates = gigacalls_per_batch / durations_s

    return ThroughputResult(
        measurement=measurement,
        aggregate=float(gigacalls_per_batch / durations_s.mean()),
        stdev=float(rates.std(ddof=1)) if rates.size > 1 else None,
        min=float(rates.min()),
        max=float(rates.max()),
    )


# ===----------------------------------------------------------------------=== #
# Launch geometry
# ===----------------------------------------------------------------------=== #


@dataclass(frozen=True, slots=True)
class _ExcludedLaunch:
    request: KernelRequest
    reason: str


class _LaunchPlanner:
    def __init__(
        self,
        device_info: DeviceInfo,
        kernel: DeviceKernel,
        request: KernelRequest,
        *,
        warps_per_block: Sequence[int],
    ) -> None:
        self.request = request
        self.unroll_factor = _get_unroll_factor(request)
        self._device_info = device_info

        try:
            self._kernel_info = kernel.info
            self.resident_blocks_by_block_size = self._find_resident_blocks(
                kernel, warps_per_block
            )
        except (RuntimeError, ValueError) as error:
            raise type(error)(
                f"{request.library} {request.symbol}: launch planning failed: {error}"
            ) from error

    def plan_launches(
        self,
        batch_sizes: tuple[int, ...],
        buffer_budget: "_LaunchBufferBudget",
        *,
        waves_per_grid: Sequence[int],
    ) -> tuple[list[ThroughputPlannedLaunch], list[_ExcludedLaunch]]:
        if not self.resident_blocks_by_block_size:
            return [], [
                _ExcludedLaunch(self.request, "no viable whole-warp block size")
            ]

        device_info = self._device_info
        launches: dict[ThroughputPlannedLaunch, None] = {}
        exclusions = []

        for waves in waves_per_grid:
            for batch_size in batch_sizes:
                for (
                    threads_per_block,
                    resident_blocks,
                ) in self.resident_blocks_by_block_size.items():
                    blocks_in_grid = min(
                        waves * device_info.multiprocessor_count * resident_blocks,
                        device_info.launch_limits.max_grid_dim[0],
                        batch_size // (threads_per_block * self.unroll_factor),
                    )
                    thread_count = blocks_in_grid * threads_per_block

                    if blocks_in_grid == 0:
                        reason = "too few input tuples for a full group per thread"
                    elif not buffer_budget.fits(batch_size, thread_count):
                        reason = (
                            "input and checksum buffers exceed "
                            f"{buffer_budget.limit_bytes} bytes"
                        )
                    else:
                        launch = ThroughputPlannedLaunch(
                            request=self.request,
                            kernel_info=self._kernel_info,
                            config=LaunchConfig(
                                grid_dim=(blocks_in_grid, 1, 1),
                                block_dim=(threads_per_block, 1, 1),
                            ),
                            max_resident_blocks_per_multiprocessor=resident_blocks,
                            batch_size=batch_size,
                        )
                        launches[launch] = None
                        continue
                    exclusions.append(
                        _ExcludedLaunch(
                            self.request,
                            f"batch_size={batch_size}, block_size={threads_per_block}, "
                            f"grid_size={blocks_in_grid}: {reason}",
                        )
                    )

        return list(launches), list(dict.fromkeys(exclusions))

    def _find_resident_blocks(
        self, kernel: DeviceKernel, warps_per_block: Sequence[int]
    ) -> dict[int, int]:
        limits = self._device_info.launch_limits
        warp_size = self._device_info.warp_size
        max_threads_per_block = min(
            limits.max_block_dim[0],
            limits.max_threads_per_block,
            self._kernel_info.max_threads_per_block,
        )
        max_warps_per_block = min(
            max(warps_per_block), max_threads_per_block // warp_size
        )
        if max_warps_per_block == 0:
            return {}

        preferred_block_sizes = dict.fromkeys(
            min(warps, max_warps_per_block) * warp_size
            for warps in sorted(warps_per_block, reverse=True)
        )
        resident_blocks_by_block_size = {}
        for threads_per_block in preferred_block_sizes:
            resident_blocks = kernel.max_active_blocks_per_multiprocessor(
                threads_per_block=threads_per_block
            )
            if resident_blocks > 0:
                resident_blocks_by_block_size[threads_per_block] = resident_blocks

        if resident_blocks_by_block_size:
            # Prefer more resident threads, then a larger block.
            return dict(
                sorted(
                    resident_blocks_by_block_size.items(),
                    key=lambda block: (block[0] * block[1], block[0]),
                    reverse=True,
                )
            )

        # If the preferred sizes fail, take the largest viable remaining block.
        for warps in range(max_warps_per_block, 0, -1):
            threads_per_block = warps * warp_size
            if threads_per_block in preferred_block_sizes:
                continue
            resident_blocks = kernel.max_active_blocks_per_multiprocessor(
                threads_per_block=threads_per_block
            )
            if resident_blocks > 0:
                return {threads_per_block: resident_blocks}
        return {}


def _get_unroll_factor(request: KernelRequest) -> int:
    unroll_factor = request.unroll_factor
    if request.kind is not KernelKind.THROUGHPUT or unroll_factor is None:
        raise ValueError(
            "expected a throughput kernel request with an unroll factor, "
            f"got kind={request.kind!r}, unroll_factor={unroll_factor!r}"
        )
    return unroll_factor


@dataclass(frozen=True, slots=True)
class _LaunchBufferBudget:
    input_bytes_per_tuple: int
    checksum_bytes_per_thread: int
    limit_bytes: int

    @classmethod
    def from_device(
        cls, spec: KernelSpec, device_info: DeviceInfo
    ) -> "_LaunchBufferBudget":
        prefix_length = len(_THROUGHPUT_PREFIX_ARGUMENTS)
        suffix_length = len(_THROUGHPUT_SUFFIX_ARGUMENTS)
        input_arguments = spec.arguments[prefix_length:-suffix_length]
        if not input_arguments:
            raise ValueError("expected input, checksum, and count arguments")
        checksums, _ = spec.arguments[-suffix_length:]

        limit_bytes = 128 << 20
        if device_info.l2_cache_size_bytes:
            limit_bytes = min(
                max(limit_bytes, 3 * device_info.l2_cache_size_bytes), 512 << 20
            )
        if device_info.total_memory_bytes > 0:
            limit_bytes = min(limit_bytes, device_info.total_memory_bytes // 16)

        return cls(
            sum(np.dtype(argument.dtype).itemsize for argument in input_arguments),
            np.dtype(checksums.dtype).itemsize,
            limit_bytes,
        )

    def fits(self, batch_size: int, thread_count: int) -> bool:
        buffer_bytes = (
            batch_size * self.input_bytes_per_tuple
            + thread_count * self.checksum_bytes_per_thread
        )
        return buffer_bytes <= self.limit_bytes


def try_plan_throughput_launch(
    device_info: DeviceInfo,
    kernel: DeviceKernel,
    request: KernelRequest,
    *,
    batch_size: int,
    threads_per_block: int,
    blocks_in_grid: int,
) -> ThroughputPlannedLaunch | None:
    limits = device_info.launch_limits
    kernel_info = kernel.info
    max_threads_per_block = min(
        kernel_info.max_threads_per_block,
        limits.max_threads_per_block,
        limits.max_block_dim[0],
    )
    thread_count = threads_per_block * blocks_in_grid

    if (
        not 1 <= batch_size <= np.iinfo(np.uint64).max
        or not 1 <= threads_per_block <= max_threads_per_block
        or not 1 <= blocks_in_grid <= limits.max_grid_dim[0]
        or thread_count * _get_unroll_factor(request) > batch_size
        or not _LaunchBufferBudget.from_device(kernel.spec, device_info).fits(
            batch_size, thread_count
        )
    ):
        return None

    max_resident_blocks_per_multiprocessor = (
        kernel.max_active_blocks_per_multiprocessor(threads_per_block=threads_per_block)
    )
    if max_resident_blocks_per_multiprocessor <= 0:
        return None
    return ThroughputPlannedLaunch(
        request=request,
        kernel_info=kernel_info,
        config=LaunchConfig(
            grid_dim=(blocks_in_grid, 1, 1),
            block_dim=(threads_per_block, 1, 1),
        ),
        max_resident_blocks_per_multiprocessor=max_resident_blocks_per_multiprocessor,
        batch_size=batch_size,
    )


def try_increase_throughput_batch(
    launch: ThroughputPlannedLaunch,
    device_info: DeviceInfo,
    kernel: DeviceKernel,
    *,
    batch_volume_growth_ratio: float,
) -> ThroughputPlannedLaunch | None:
    if not isfinite(batch_volume_growth_ratio) or batch_volume_growth_ratio <= 0:
        raise ValueError("batch_volume_growth_ratio must be finite and positive")

    budget = _LaunchBufferBudget.from_device(kernel.spec, device_info)
    max_batch_size = (
        budget.limit_bytes - launch.thread_count * budget.checksum_bytes_per_thread
    ) // budget.input_bytes_per_tuple
    available_growth = max_batch_size - launch.batch_size
    if available_growth <= 0:
        return None

    growth = max(
        1, int(min(available_growth, launch.batch_size * batch_volume_growth_ratio))
    )
    return replace(launch, batch_size=launch.batch_size + growth)


# ===----------------------------------------------------------------------=== #
# Search planning
# ===----------------------------------------------------------------------=== #


@dataclass(frozen=True, slots=True)
class ThroughputSearchPlan:
    candidates: tuple[ThroughputPlannedLaunch, ...]
    eligible: tuple[ThroughputPlannedLaunch, ...]
    batch_source: str
    exclusions: tuple[_ExcludedLaunch, ...]


def plan_throughput_search(
    device_info: DeviceInfo,
    kernels: Mapping[KernelRequest, DeviceKernel],
    *,
    search: ThroughputSearchConfig,
) -> ThroughputSearchPlan:
    max_candidates = _normalize_integer(
        search.max_candidates, "max_candidates", minimum=4
    )
    for name, values in (
        ("batch_volume_to_l2_ratios", search.batch_volume_to_l2_ratios),
        ("fallback_batch_volumes_mib", search.fallback_batch_volumes_mib),
        ("warps_per_block", search.warps_per_block),
        ("waves_per_grid", search.waves_per_grid),
    ):
        if (
            not values
            or any(
                value <= 0 or (isinstance(value, float) and not isfinite(value))
                for value in values
            )
            or len(set(values)) != len(values)
        ):
            raise ValueError(f"{name} must contain distinct, finite, positive values")

    if not kernels:
        raise ValueError("expected at least one throughput kernel")

    requests_by_unroll = sorted(kernels, key=_get_unroll_factor)
    first_request = requests_by_unroll[0]
    spec = kernels[first_request].spec
    if any(
        request.library != first_request.library
        or request.function != first_request.function
        or kernels[request].spec != spec
        for request in requests_by_unroll
    ):
        raise ValueError("throughput search kernels must share a library and signature")

    buffer_budget = _LaunchBufferBudget.from_device(spec, device_info)
    batch_sizes, batch_source = _choose_search_batch_sizes(
        device_info,
        buffer_budget,
        batch_volume_to_l2_ratios=search.batch_volume_to_l2_ratios,
        fallback_batch_volumes_mib=search.fallback_batch_volumes_mib,
    )

    launches_by_kernel = []
    exclusions = []
    for request in requests_by_unroll:
        planner = _LaunchPlanner(
            device_info,
            kernels[request],
            request,
            warps_per_block=search.warps_per_block,
        )
        launches, rejected_launches = planner.plan_launches(
            batch_sizes, buffer_budget, waves_per_grid=search.waves_per_grid
        )
        launches_by_kernel.append(launches)
        exclusions.extend(rejected_launches)

    # Give every viable unroll factor a turn before taking its next launch.
    eligible = tuple(
        launch
        for launches in zip_longest(*launches_by_kernel)
        for launch in launches
        if launch is not None
    )
    return ThroughputSearchPlan(
        candidates=eligible[:max_candidates],
        eligible=eligible,
        batch_source=batch_source,
        exclusions=tuple(exclusions),
    )


def _choose_search_batch_sizes(
    device_info: DeviceInfo,
    buffer_budget: _LaunchBufferBudget,
    *,
    batch_volume_to_l2_ratios: Sequence[float],
    fallback_batch_volumes_mib: Sequence[int],
) -> tuple[tuple[int, ...], str]:
    l2_bytes = device_info.l2_cache_size_bytes
    if l2_bytes:
        volumes = tuple(l2_bytes * ratio for ratio in batch_volume_to_l2_ratios)
        source = "L2 cache size"
    else:
        volumes = tuple(volume << 20 for volume in fallback_batch_volumes_mib)
        source = "fallback batch volumes (L2 cache size unavailable)"

    # Reserve space for checksums, which are checked against the actual grid.
    max_input_bytes = buffer_budget.limit_bytes * 15 // 16
    batch_sizes = dict.fromkeys(
        max(1, int(min(volume, max_input_bytes)) // buffer_budget.input_bytes_per_tuple)
        for volume in volumes
    )
    return tuple(batch_sizes), source


# ===----------------------------------------------------------------------=== #
# Shared validation
# ===----------------------------------------------------------------------=== #


def _normalize_integer(value: int, name: str, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    try:
        value = index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer") from None
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value
