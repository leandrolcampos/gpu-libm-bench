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

"""Dependent-call latency measurement and analysis."""

from dataclasses import dataclass
from math import inf, isfinite
from operator import index
from time import monotonic, sleep
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from gpu_libm_bench import inputs
from gpu_libm_bench.devices import (
    DeviceBuffer,
    DeviceContext,
    DeviceInfo,
    DeviceKernel,
    KernelMetric,
    LaunchConfig,
)
from gpu_libm_bench.types import Scalar

from ._throttle import recovery_delay_s

_PASS_COUNT = 2
_LATENCY_ARGUMENTS = (
    ("pass_count", False, "uint32"),
    ("chain_length", False, "uint32"),
    ("mask", True, None),
    ("elapsed_ticks", True, "uint64"),
    ("output", True, None),
)

_CHAIN_COST_LENGTHS = (8192, 32768, 131072)
_CHAIN_COST_SAMPLES_PER_LENGTH = 5
_CHAIN_COST_TIMEOUT_S = 2.0
_CHAIN_COST_MAX_RELATIVE_SLOPE_DIFFERENCE = 0.05


@dataclass(frozen=True, slots=True)
class LatencyMeasurement:
    inputs: inputs.InputBatch
    elapsed_ticks: NDArray[np.uint64]
    device_info: DeviceInfo
    pass_count: int
    chain_length: int
    warmup_runs: int
    clock_frequencies_hz: NDArray[np.float64] | None = None


@dataclass(frozen=True, slots=True)
class LatencyResult:
    measurement: LatencyMeasurement
    latencies: NDArray[np.float64]
    mean: float
    stdev: float | None
    min: float
    max: float
    mean_chain_cost_discount: float | None = None
    discount_skip_reason: str | None = None
    unit: ClassVar[str] = "ns/call"


@dataclass(frozen=True, slots=True)
class ChainCostEstimate:
    lengths: tuple[int, ...]
    cycles_per_call: float | None = None
    reason: str | None = None


@dataclass(slots=True)
class _LatencyBuffers:
    inputs: tuple[DeviceBuffer[Scalar], ...]
    mask: DeviceBuffer[Scalar]
    elapsed_ticks: DeviceBuffer[np.uint64]
    output: DeviceBuffer[Scalar]
    host_elapsed_ticks: NDArray[np.uint64]


class LatencyBenchmark:
    def __init__(
        self,
        context: DeviceContext,
        kernel: DeviceKernel,
        *,
        timeout_s: float | None = None,
        chain_length: int,
        warmup_runs: int,
        min_clock_ratio: float = 0.0,
        initial_delay_s: float = 0.0,
    ) -> None:
        arguments = iter(kernel.spec.arguments)
        latency_dtypes = {}
        for name, is_pointer, dtype in _LATENCY_ARGUMENTS:
            argument = next(arguments, None)
            if (
                argument is None
                or argument.name != name
                or argument.is_pointer != is_pointer
            ):
                raise ValueError("kernel does not match the latency protocol")
            actual_dtype = np.dtype(argument.dtype)
            if dtype is not None and actual_dtype != np.dtype(dtype):
                raise ValueError(f"{name} must have dtype {dtype}")
            latency_dtypes[name] = actual_dtype

        input_arguments = tuple(arguments)
        if not input_arguments:
            raise ValueError("latency kernel must declare at least one input argument")
        if not all(argument.is_pointer for argument in input_arguments):
            raise ValueError("latency kernel input arguments must be pointers")
        first_input_dtype = np.dtype(input_arguments[0].dtype)
        if first_input_dtype.kind != "f" or first_input_dtype.itemsize not in (2, 4, 8):
            raise ValueError(
                "latency kernel first input must have a supported float dtype"
            )
        storage_dtype = np.dtype(f"uint{8 * first_input_dtype.itemsize}")
        for name in ("mask", "output"):
            if latency_dtypes[name] != storage_dtype:
                raise ValueError(f"{name} must have dtype {storage_dtype}")

        if timeout_s is not None and (not isfinite(timeout_s) or timeout_s <= 0):
            raise ValueError("timeout_s must be finite and positive")

        chain_length = _normalize_integer(chain_length, "chain_length")
        max_chain_length = np.iinfo(latency_dtypes["chain_length"]).max
        if not 1 <= chain_length <= max_chain_length:
            raise ValueError(f"chain_length must be between 1 and {max_chain_length}")

        warmup_runs = _normalize_integer(warmup_runs, "warmup_runs")
        if warmup_runs < 0:
            raise ValueError("warmup_runs must be nonnegative")

        if not isfinite(min_clock_ratio) or min_clock_ratio < 0:
            raise ValueError("min_clock_ratio must be finite and nonnegative")
        if not isfinite(initial_delay_s) or initial_delay_s < 0:
            raise ValueError("initial_delay_s must be finite and nonnegative")

        self._context = context
        self._kernel = kernel
        self._latency_dtypes = latency_dtypes
        self._input_arguments = input_arguments

        self._timeout_s = timeout_s
        self._pass_count = latency_dtypes["pass_count"].type(_PASS_COUNT)
        self._chain_length = latency_dtypes["chain_length"].type(chain_length)
        self._warmup_runs = warmup_runs
        self._min_clock_ratio = min_clock_ratio
        self._initial_delay_s = initial_delay_s

    def measure(
        self, batch: inputs.InputBatch, *, record_clocks: bool = False
    ) -> LatencyMeasurement:
        snapshot = self._snapshot_batch(batch)
        clocks_hz = (
            np.empty(snapshot.size)
            if record_clocks or self._min_clock_ratio > 0
            else None
        )
        elapsed_ticks = self._collect_ticks(
            snapshot, self._chain_length, clocks_hz=clocks_hz
        )
        return LatencyMeasurement(
            inputs=snapshot,
            elapsed_ticks=elapsed_ticks,
            device_info=self._context.device_info,
            pass_count=int(self._pass_count),
            chain_length=int(self._chain_length),
            warmup_runs=self._warmup_runs,
            clock_frequencies_hz=clocks_hz,
        )

    def estimate_chain_cost(self) -> ChainCostEstimate:
        lengths = _CHAIN_COST_LENGTHS
        deadline = monotonic() + _CHAIN_COST_TIMEOUT_S
        (argument,) = self._input_arguments
        batch = inputs.InputBatch(
            "identity",
            {
                argument.name: np.zeros(
                    _CHAIN_COST_SAMPLES_PER_LENGTH, dtype=argument.dtype
                )
            },
        )
        timer_hz = self._context.device_info.timer_info.frequency_hz
        clocks_hz = np.empty(batch.size, dtype=np.float64)
        cycles = np.empty((len(lengths), batch.size), dtype=np.float64)
        resolution_cycles = 0.0

        for row, length in enumerate(lengths):
            try:
                ticks = self._collect_ticks(
                    batch,
                    length,
                    deadline=deadline,
                    clocks_hz=clocks_hz,
                    require_clock=True,
                )
            except _SampleTimeout as error:
                return ChainCostEstimate(lengths, reason=str(error))

            cycles_per_tick = clocks_hz / timer_hz
            cycles[row] = ticks * cycles_per_tick
            resolution_cycles = max(resolution_cycles, float(cycles_per_tick.max()))

        return _fit_chain_cost(lengths, cycles, resolution_cycles)

    def _collect_ticks(
        self,
        batch: inputs.InputBatch,
        length: int,
        *,
        deadline: float = inf,
        clocks_hz: NDArray[np.float64] | None = None,
        require_clock: bool = False,
    ) -> NDArray[np.uint64]:
        input_arrays = tuple(
            batch.arguments[input_argument.name]
            for input_argument in self._input_arguments
        )

        min_clock_hz = self._minimum_clock_hz()
        profile_clock = clocks_hz is not None or min_clock_hz > 0

        elapsed_ticks = np.empty(
            batch.size,
            dtype=self._latency_dtypes["elapsed_ticks"],
        )
        buffers = self._create_buffers()

        for run in range(self._warmup_runs):
            if deadline != inf:
                _remaining_sample_s(deadline, batch, 0, "warmup timed out")
            self._copy_inputs_to_device(input_arrays, buffers, run % batch.size)
            self._run_chain(buffers, length=length)
            if deadline != inf:
                self._context.copy_to_host(
                    buffers.host_elapsed_ticks, buffers.elapsed_ticks
                )

        if deadline == inf and self._timeout_s is not None:
            if self._warmup_runs:
                self._context.copy_to_host(
                    buffers.host_elapsed_ticks, buffers.elapsed_ticks
                )
            deadline = monotonic() + self._timeout_s

        delay_s = self._initial_delay_s
        for input_index in range(batch.size):
            self._copy_inputs_to_device(input_arrays, buffers, input_index)

            consecutive_discards = 0
            reason = "insufficient observations"
            while True:
                _remaining_sample_s(deadline, batch, input_index, reason)
                clock_hz = self._run_chain(
                    buffers, length=length, profile_clock=profile_clock
                )
                if (require_clock or min_clock_hz > 0) and (
                    clock_hz is None or not isfinite(clock_hz) or clock_hz <= 0
                ):
                    reason = "clock unavailable"
                elif (
                    min_clock_hz > 0
                    and clock_hz is not None
                    and clock_hz < min_clock_hz
                ):
                    reason = "clock below threshold"
                else:
                    self._context.copy_to_host(
                        buffers.host_elapsed_ticks, buffers.elapsed_ticks
                    )
                    _remaining_sample_s(
                        deadline, batch, input_index, "insufficient observations"
                    )
                    elapsed_ticks[input_index] = buffers.host_elapsed_ticks[0]
                    if clocks_hz is not None:
                        clocks_hz[input_index] = clock_hz
                    break

                remaining_s = _remaining_sample_s(deadline, batch, input_index, reason)
                consecutive_discards += 1
                delay_s = recovery_delay_s(delay_s, consecutive_discards)

                if delay_s > 0:
                    sleep(min(delay_s, remaining_s))

        return elapsed_ticks

    def _snapshot_batch(self, batch: inputs.InputBatch) -> inputs.InputBatch:
        if set(batch.arguments) != {
            input_argument.name for input_argument in self._input_arguments
        }:
            raise ValueError("latency inputs must match the kernel input arguments")

        size: int | None = None
        for input_argument in self._input_arguments:
            source = batch.arguments[input_argument.name]
            label = f"latency input {input_argument.name!r}"

            if not isinstance(source, np.ndarray):
                raise TypeError(f"{label} must be a NumPy array")
            if source.dtype != np.dtype(input_argument.dtype):
                raise ValueError(f"{label} must have dtype {input_argument.dtype}")
            if source.ndim != 1 or source.size == 0:
                raise ValueError(f"{label} must be a nonempty one-dimensional array")

            if size is not None and source.size != size:
                raise ValueError("latency inputs must have the same number of elements")
            size = source.size

        return inputs.InputBatch(
            batch.domain,
            {
                input_argument.name: batch.arguments[input_argument.name].copy()
                for input_argument in self._input_arguments
            },
        )

    def _minimum_clock_hz(self) -> float:
        if self._min_clock_ratio == 0:
            return 0.0
        default = self._context.device_info.default_clock_frequency_hz
        if default <= 0:
            raise ValueError(
                "throttle detection requires a positive default clock frequency"
            )
        return self._min_clock_ratio * default

    def _create_buffers(self) -> _LatencyBuffers:
        latency_dtypes = self._latency_dtypes
        buffers = _LatencyBuffers(
            inputs=tuple(
                self._context.create_buffer(1, dtype=input_argument.dtype)
                for input_argument in self._input_arguments
            ),
            mask=self._context.create_buffer(1, dtype=latency_dtypes["mask"]),
            elapsed_ticks=self._context.create_buffer(
                1, dtype=latency_dtypes["elapsed_ticks"]
            ),
            output=self._context.create_buffer(1, dtype=latency_dtypes["output"]),
            host_elapsed_ticks=np.empty(1, dtype=latency_dtypes["elapsed_ticks"]),
        )

        self._context.copy_to_device(
            buffers.mask, np.zeros(1, dtype=latency_dtypes["mask"])
        )
        return buffers

    def _copy_inputs_to_device(
        self,
        input_arrays: tuple[NDArray[Scalar], ...],
        buffers: _LatencyBuffers,
        input_index: int,
    ) -> None:
        for destination, source in zip(buffers.inputs, input_arrays, strict=True):
            self._context.copy_to_device(
                destination, source[input_index : input_index + 1]
            )

    def _run_chain(
        self,
        buffers: _LatencyBuffers,
        *,
        length: int,
        profile_clock: bool = False,
    ) -> float | None:
        arguments = (
            self._pass_count,
            np.uint32(length),
            buffers.mask,
            buffers.elapsed_ticks,
            buffers.output,
            *buffers.inputs,
        )
        if profile_clock:
            profile = self._context.launch_and_profile(
                self._kernel,
                *arguments,
                config=LaunchConfig(),
                metrics={KernelMetric.CLOCK_FREQUENCY},
            )
            return profile.clock_frequency_hz

        self._context.launch(self._kernel, *arguments, config=LaunchConfig())
        return None


def _fit_chain_cost(
    lengths: tuple[int, ...], cycles: NDArray[np.float64], resolution_cycles: float
) -> ChainCostEstimate:
    medians = np.median(cycles, axis=1)
    deviations = np.median(np.abs(cycles - medians[:, None]), axis=1)
    growth = np.diff(medians)
    if np.any(growth <= resolution_cycles + deviations[:-1] + deviations[1:]):
        return ChainCostEstimate(lengths, reason="growth indistinguishable from noise")

    slopes = growth / np.diff(lengths)
    if np.ptp(slopes) > _CHAIN_COST_MAX_RELATIVE_SLOPE_DIFFERENCE * np.mean(slopes):
        return ChainCostEstimate(lengths, reason="inconsistent interval slopes")

    cycles_per_call, _ = np.polyfit(lengths, medians, deg=1)
    return ChainCostEstimate(lengths, cycles_per_call=float(cycles_per_call))


def analyze(
    measurement: LatencyMeasurement,
    *,
    chain_cost: ChainCostEstimate | None = None,
    should_discount_chain_cost: bool = True,
) -> LatencyResult:
    times = measurement.device_info.timer_info.to_ns(measurement.elapsed_ticks)
    latencies = times / measurement.chain_length

    mean_chain_cost_discount = None
    discount_skip_reason = None
    if not should_discount_chain_cost:
        discount_skip_reason = "discount disabled"
    elif chain_cost is not None:
        clocks = measurement.clock_frequencies_hz
        if chain_cost.cycles_per_call is None:
            discount_skip_reason = "calibration inconclusive"
        elif (
            clocks is None
            or clocks.shape != latencies.shape
            or not np.all(np.isfinite(clocks) & (clocks > 0))
        ):
            discount_skip_reason = "incomplete launch clocks"
        else:
            discounts_ns = chain_cost.cycles_per_call * 1e9 / clocks
            latencies -= discounts_ns
            mean_chain_cost_discount = float(np.mean(discounts_ns))

    return LatencyResult(
        measurement=measurement,
        latencies=latencies,
        mean=float(np.mean(latencies)),
        stdev=float(np.std(latencies, ddof=1)) if latencies.size > 1 else None,
        min=float(np.min(latencies)),
        max=float(np.max(latencies)),
        mean_chain_cost_discount=mean_chain_cost_discount,
        discount_skip_reason=discount_skip_reason,
    )


def _normalize_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")

    try:
        return index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error


class _SampleTimeout(TimeoutError):
    pass


def _remaining_sample_s(
    deadline: float, batch: inputs.InputBatch, input_index: int, reason: str
) -> float:
    remaining_s = deadline - monotonic()
    if remaining_s <= 0:
        raise _SampleTimeout(
            f"latency sample for domain {batch.domain!r} is incomplete after timeout: "
            f"{input_index}/{batch.size} observations accepted, "
            f"input index {input_index} pending ({reason})"
        )
    return remaining_s
