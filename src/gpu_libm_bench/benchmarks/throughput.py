# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Independent-call throughput launch planning, measurement, and analysis."""

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from fractions import Fraction
from itertools import chain, zip_longest
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

# ===----------------------------------------------------------------------=== #
# Launch geometry
# ===----------------------------------------------------------------------=== #

_PREFERRED_WARPS_PER_BLOCK = (8, 4)
_TARGET_GRID_WAVES = (4, 2, 1)


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
    batch_size = _normalize_integer(batch_size, "batch_size", minimum=1)
    max_batch_size = np.iinfo(np.uint64).max
    if batch_size > max_batch_size:
        raise ValueError(
            f"batch_size={batch_size} exceeds the maximum supported batch size "
            f"of {max_batch_size} input tuples per launch"
        )

    planner = _KernelLaunchPlanner(device_info, kernel, request)
    if not planner.block_candidates:
        return ThroughputExcludedLaunch(request, "no viable whole-warp block size")

    for launch_candidate in planner.iter_launch_candidates(batch_size=batch_size):
        if (
            planner.unroll_factor == 1
            or launch_candidate.min_unrolled_groups_per_thread > 0
        ):
            return launch_candidate.launch
    return ThroughputExcludedLaunch(
        request,
        f"batch_size={batch_size} cannot give each thread {planner.unroll_factor} "
        "evaluations at any candidate launch geometry",
    )


@dataclass(frozen=True, slots=True)
class _BlockOccupancy:
    threads_per_block: int
    max_resident_blocks_per_multiprocessor: int


@dataclass(frozen=True, slots=True)
class _LaunchCandidate:
    launch: ThroughputPlannedLaunch
    target_waves: int
    min_unrolled_groups_per_thread: int


class _KernelLaunchPlanner:
    def __init__(
        self, device_info: DeviceInfo, kernel: DeviceKernel, request: KernelRequest
    ) -> None:
        self.request = request
        self.unroll_factor = _get_unroll_factor(request)
        self._device_info = device_info

        try:
            self._kernel_info = kernel.info
            self.block_candidates = self._select_block_candidates(kernel)
        except (RuntimeError, ValueError) as error:
            raise type(error)(
                f"{request.library} {request.symbol}: launch planning failed: {error}"
            ) from error

    def iter_launch_candidates(self, *, batch_size: int) -> Iterator[_LaunchCandidate]:
        device_info = self._device_info
        for block in self.block_candidates:
            threads_per_block = block.threads_per_block
            blocks_per_wave = (
                device_info.multiprocessor_count
                * block.max_resident_blocks_per_multiprocessor
            )
            blocks_to_cover_batch = (
                batch_size + threads_per_block - 1
            ) // threads_per_block
            max_grid_blocks = min(
                blocks_to_cover_batch, device_info.launch_limits.max_grid_dim[0]
            )

            for target_waves in _TARGET_GRID_WAVES:
                grid_blocks = min(target_waves * blocks_per_wave, max_grid_blocks)
                launch = ThroughputPlannedLaunch(
                    request=self.request,
                    kernel_info=self._kernel_info,
                    config=LaunchConfig(
                        grid_dim=(grid_blocks, 1, 1),
                        block_dim=(threads_per_block, 1, 1),
                    ),
                    max_resident_blocks_per_multiprocessor=block.max_resident_blocks_per_multiprocessor,
                    batch_size=batch_size,
                )
                yield _LaunchCandidate(
                    launch=launch,
                    target_waves=target_waves,
                    min_unrolled_groups_per_thread=batch_size
                    // (launch.thread_count * self.unroll_factor),
                )

    def _select_block_candidates(self, kernel: DeviceKernel) -> list[_BlockOccupancy]:
        limits = self._device_info.launch_limits
        warp_size = self._device_info.warp_size
        max_threads_per_block = min(
            limits.max_block_dim[0],
            limits.max_threads_per_block,
            self._kernel_info.max_threads_per_block,
        )
        max_warps_per_block = min(
            max(_PREFERRED_WARPS_PER_BLOCK), max_threads_per_block // warp_size
        )
        if max_warps_per_block == 0:
            return []

        preferred_block_sizes = dict.fromkeys(
            min(warps, max_warps_per_block) * warp_size
            for warps in _PREFERRED_WARPS_PER_BLOCK
        )
        block_candidates = []
        for threads_per_block in preferred_block_sizes:
            resident_blocks = kernel.max_active_blocks_per_multiprocessor(
                threads_per_block=threads_per_block
            )
            if resident_blocks > 0:
                block_candidates.append(
                    _BlockOccupancy(threads_per_block, resident_blocks)
                )

        if block_candidates:
            # Prefer more resident threads, then a larger block.
            return sorted(
                block_candidates,
                key=lambda block: (
                    block.threads_per_block
                    * block.max_resident_blocks_per_multiprocessor,
                    block.threads_per_block,
                ),
                reverse=True,
            )

        # If both preferred sizes fail, take the largest viable remaining block.
        for warps_per_block in range(max_warps_per_block, 0, -1):
            threads_per_block = warps_per_block * warp_size
            if threads_per_block in preferred_block_sizes:
                continue
            resident_blocks = kernel.max_active_blocks_per_multiprocessor(
                threads_per_block=threads_per_block
            )
            if resident_blocks > 0:
                return [_BlockOccupancy(threads_per_block, resident_blocks)]
        return []


def _get_unroll_factor(request: KernelRequest) -> int:
    unroll_factor = request.unroll_factor
    if request.kind is not KernelKind.THROUGHPUT or unroll_factor is None:
        raise ValueError(
            "expected a throughput kernel request with an unroll factor, "
            f"got kind={request.kind!r}, unroll_factor={unroll_factor!r}"
        )
    return unroll_factor


# ===----------------------------------------------------------------------=== #
# Search planning
# ===----------------------------------------------------------------------=== #

_L2_BATCH_RATIOS = ((2, 3), (1, 3), (1, 1), (4, 3))
_FALLBACK_BATCH_VOLUMES_MIB = (32, 16, 4, 64)
_MAX_BUFFER_BYTES = 128 << 20


@dataclass(frozen=True, slots=True)
class ThroughputSearchPlan:
    candidates: tuple[ThroughputPlannedLaunch, ...]
    eligible: tuple[ThroughputPlannedLaunch, ...]
    batch_source: str
    exclusions: tuple[ThroughputExcludedLaunch, ...]


def plan_throughput_search(
    device_info: DeviceInfo,
    kernels: Mapping[KernelRequest, DeviceKernel],
    *,
    max_candidates: int,
) -> ThroughputSearchPlan:
    max_candidates = _normalize_integer(max_candidates, "max_candidates", minimum=4)
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
    if len(spec.arguments) < 3:
        raise ValueError("expected input, checksum, and count arguments")

    input_bytes_per_tuple = sum(
        np.dtype(argument.dtype).itemsize for argument in spec.arguments[:-2]
    )
    checksum_bytes_per_thread = np.dtype(spec.arguments[-2].dtype).itemsize
    batch_sizes, batch_source = _choose_search_batch_sizes(
        device_info.l2_cache_size_bytes, input_bytes_per_tuple
    )
    buffer_limit_bytes = _MAX_BUFFER_BYTES
    if device_info.total_memory_bytes > 0:
        buffer_limit_bytes = min(
            buffer_limit_bytes, device_info.total_memory_bytes // 16
        )

    planners = []
    exclusions = []
    for request in requests_by_unroll:
        planner = _KernelLaunchPlanner(device_info, kernels[request], request)
        if planner.block_candidates:
            planners.append(planner)
        else:
            exclusions.append(
                ThroughputExcludedLaunch(request, "no viable whole-warp block size")
            )

    launches_by_batch_and_kernel = []
    for batch_size in batch_sizes:
        input_buffer_bytes = batch_size * input_bytes_per_tuple
        for planner in planners:
            viable_candidates, rejected_launches = _filter_launch_candidates(
                planner,
                batch_size=batch_size,
                input_buffer_bytes=input_buffer_bytes,
                checksum_bytes_per_thread=checksum_bytes_per_thread,
                buffer_limit_bytes=buffer_limit_bytes,
            )
            if viable_candidates:
                viable_candidates.sort(key=_get_launch_priority, reverse=True)
                launches_by_batch_and_kernel.append(
                    [candidate.launch for candidate in viable_candidates]
                )
            exclusions.extend(rejected_launches)

    eligible = _order_search_launches(requests_by_unroll, launches_by_batch_and_kernel)
    return ThroughputSearchPlan(
        candidates=eligible[:max_candidates],
        eligible=eligible,
        batch_source=batch_source,
        exclusions=tuple(exclusions),
    )


def _choose_search_batch_sizes(
    l2_cache_size_bytes: int, input_bytes_per_tuple: int
) -> tuple[tuple[int, ...], str]:
    if l2_cache_size_bytes:
        volumes_bytes = (
            l2_cache_size_bytes * numerator // denominator
            for numerator, denominator in _L2_BATCH_RATIOS
        )
        source = "L2 cache size"
    else:
        volumes_bytes = (volume << 20 for volume in _FALLBACK_BATCH_VOLUMES_MIB)
        source = "device has no L2 cache"
    batch_sizes = tuple(
        dict.fromkeys(
            max(1, volume // input_bytes_per_tuple) for volume in volumes_bytes
        )
    )
    return batch_sizes, source


def _filter_launch_candidates(
    planner: _KernelLaunchPlanner,
    *,
    batch_size: int,
    input_buffer_bytes: int,
    checksum_bytes_per_thread: int,
    buffer_limit_bytes: int,
) -> tuple[list[_LaunchCandidate], list[ThroughputExcludedLaunch]]:
    unique_geometries: dict[LaunchConfig, _LaunchCandidate] = {}
    for launch_candidate in planner.iter_launch_candidates(batch_size=batch_size):
        config = launch_candidate.launch.config
        # Keep the two-wave target when grid limits collapse multiple targets.
        if launch_candidate.target_waves == 2 or config not in unique_geometries:
            unique_geometries[config] = launch_candidate

    viable_launch_candidates = []
    exclusions = []
    for launch_candidate in unique_geometries.values():
        launch = launch_candidate.launch
        buffer_bytes = (
            input_buffer_bytes + launch.thread_count * checksum_bytes_per_thread
        )
        if (
            planner.unroll_factor > 1
            and launch_candidate.min_unrolled_groups_per_thread == 0
        ):
            reason = "too few input tuples for a full group per thread"
        elif buffer_bytes > buffer_limit_bytes:
            reason = f"input and checksum buffers exceed {buffer_limit_bytes} bytes"
        else:
            viable_launch_candidates.append(launch_candidate)
            continue
        exclusions.append(
            ThroughputExcludedLaunch(
                planner.request,
                f"batch_size={batch_size}, "
                f"block_size={launch.config.block_dim[0]}, "
                f"grid_size={launch.config.grid_dim[0]}: {reason}",
            )
        )

    return viable_launch_candidates, exclusions


def _get_launch_priority(
    launch_candidate: _LaunchCandidate,
) -> tuple[int, bool, int, int]:
    threads_per_block = launch_candidate.launch.config.block_dim[0]
    resident_threads = (
        threads_per_block
        * launch_candidate.launch.max_resident_blocks_per_multiprocessor
    )
    return (
        resident_threads,
        launch_candidate.target_waves == 2,
        launch_candidate.min_unrolled_groups_per_thread,
        threads_per_block,
    )


def _order_search_launches(
    requests_by_unroll: list[KernelRequest],
    launches_by_batch_and_kernel: list[list[ThroughputPlannedLaunch]],
) -> tuple[ThroughputPlannedLaunch, ...]:
    first_launch_by_kernel: dict[KernelRequest, ThroughputPlannedLaunch] = {}
    for ranked_launches in launches_by_batch_and_kernel:
        preferred_launch = ranked_launches[0]
        first_launch_by_kernel.setdefault(preferred_launch.request, preferred_launch)

    # Give every viable kernel a slot, then cover the other batch sizes.
    first_launches = tuple(
        first_launch_by_kernel[request]
        for request in requests_by_unroll
        if request in first_launch_by_kernel
    )
    other_batch_leaders = tuple(
        ranked_launches[0]
        for ranked_launches in launches_by_batch_and_kernel
        if ranked_launches[0] not in first_launches
    )

    # Offer the first block and grid contrasts before round-robin extras.
    contrasts_by_batch_and_kernel = [
        _order_geometry_contrasts(ranked_launches)
        for ranked_launches in launches_by_batch_and_kernel
    ]
    first_contrasts = tuple(
        launch
        for contrasts in contrasts_by_batch_and_kernel
        for launch in contrasts[:2]
    )
    remaining_contrasts = tuple(
        launch
        for round_of_contrasts in zip_longest(
            *(contrasts[2:] for contrasts in contrasts_by_batch_and_kernel)
        )
        for launch in round_of_contrasts
        if launch is not None
    )
    return first_launches + other_batch_leaders + first_contrasts + remaining_contrasts


def _order_geometry_contrasts(
    ranked_launches: list[ThroughputPlannedLaunch],
) -> list[ThroughputPlannedLaunch]:
    preferred_launch, *alternatives = ranked_launches
    different_block_size = [
        launch
        for launch in alternatives
        if launch.config.block_dim != preferred_launch.config.block_dim
    ]
    same_block_size = [
        launch
        for launch in alternatives
        if launch.config.block_dim == preferred_launch.config.block_dim
    ]
    preferred_grid_to_residency_ratio = Fraction(
        preferred_launch.config.grid_dim[0],
        preferred_launch.max_resident_blocks_per_multiprocessor,
    )
    different_block_size.sort(
        key=lambda launch: abs(
            Fraction(
                launch.config.grid_dim[0], launch.max_resident_blocks_per_multiprocessor
            )
            - preferred_grid_to_residency_ratio
        )
    )
    return (
        different_block_size[:1]
        + same_block_size[:1]
        + different_block_size[1:]
        + same_block_size[1:]
    )


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

        sample_size = _normalize_integer(sample_size, "sample_size", minimum=1)
        warmup_runs = _normalize_integer(warmup_runs, "warmup_runs", minimum=0)
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
            launch.thread_count,
            dtype=self._checksum_dtype,
        )
        arguments: tuple[DeviceBuffer[Any] | Scalar, ...] = (
            *input_buffers,
            checksums,
            np.uint64(launch.batch_size),
        )
        batch_iterator = iter(batches)
        if self._warmup_runs:
            first_batch = _get_next_batch(batch_iterator, domain, 0)
            self._copy_inputs_to_device(
                input_buffers, self._prepare_input_arrays(first_batch, domain)
            )
            for _ in range(self._warmup_runs):
                context.launch(self._kernel, *arguments, config=launch.config)
            batch_iterator = chain((first_batch,), batch_iterator)

        deadline = inf if self._timeout_s is None else monotonic() + self._timeout_s
        attempts: list[ThroughputAttempt] = []
        timed_out = self._collect_attempts(
            domain,
            batch_iterator,
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

    def _collect_attempts(
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
                reason = _get_discard_reason(clock_hz, clock_threshold, remaining_s)
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

    def _copy_inputs_to_device(
        self,
        buffers: tuple[DeviceBuffer[Scalar], ...],
        arrays: tuple[NDArray[Scalar], ...],
    ) -> None:
        for buffer, array in zip(buffers, arrays, strict=True):
            self._context.copy_to_device(buffer, array)


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


# ===----------------------------------------------------------------------=== #
# Analysis
# ===----------------------------------------------------------------------=== #


@dataclass(frozen=True, slots=True)
class ThroughputResult:
    measurement: ThroughputMeasurement
    mean: float
    stdev: float | None
    min: float
    max: float
    unit: ClassVar[str] = "Gcall/s"


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
    gigacalls_per_batch = measurement.launch.batch_size / 1e9
    durations_s = durations_ms * 1e-3
    rates = gigacalls_per_batch / durations_s
    return ThroughputResult(
        measurement=measurement,
        mean=float(gigacalls_per_batch / durations_s.mean()),
        stdev=float(rates.std(ddof=1)) if rates.size > 1 else None,
        min=float(rates.min()),
        max=float(rates.max()),
    )


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
