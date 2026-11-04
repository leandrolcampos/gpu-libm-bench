# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Dependent-call latency measurement and analysis."""

from dataclasses import dataclass
from operator import index
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from gpu_libm_bench import inputs
from gpu_libm_bench.devices import (
    DeviceBuffer,
    DeviceContext,
    DeviceInfo,
    DeviceKernel,
    LaunchConfig,
)
from gpu_libm_bench.statistics import compute_mean_confidence_interval
from gpu_libm_bench.types import Scalar

_PASS_COUNT = 2
_LATENCY_ARGUMENT_NAMES = (
    "pass_count",
    "chain_length",
    "mask",
    "elapsed_ticks",
    "output",
)


@dataclass(frozen=True, slots=True)
class LatencyMeasurement:
    inputs: inputs.InputBatch
    elapsed_ticks: NDArray[np.uint64]
    device_info: DeviceInfo
    pass_count: int
    chain_length: int
    warmup_runs: int


@dataclass(frozen=True, slots=True)
class LatencyResult:
    measurement: LatencyMeasurement
    latencies: NDArray[np.float64]
    mean: float
    mean_confidence_interval: tuple[float, float]
    confidence_level: float
    bootstrap_resamples: int
    stdev: float | None
    min: float
    max: float
    unit: ClassVar[str] = "ns/call"


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
        chain_length: int,
        repetitions: int,
        warmup_runs: int,
    ) -> None:
        arguments = iter(kernel.spec.arguments)
        latency_dtypes = {}
        for expected_name in _LATENCY_ARGUMENT_NAMES:
            argument = next(arguments, None)
            if argument is None or argument.name != expected_name:
                raise ValueError("kernel does not match the latency protocol")
            latency_dtypes[expected_name] = np.dtype(argument.dtype)

        if latency_dtypes["elapsed_ticks"] != np.dtype(np.uint64):
            raise ValueError("elapsed_ticks must have dtype uint64")

        input_arguments = tuple(arguments)
        if not input_arguments:
            raise ValueError("latency kernel must declare at least one input argument")

        chain_length = _as_integer(chain_length, "chain_length")
        max_chain_length = np.iinfo(latency_dtypes["chain_length"]).max
        if not 1 <= chain_length <= max_chain_length:
            raise ValueError(f"chain_length must be between 1 and {max_chain_length}")

        repetitions = _as_integer(repetitions, "repetitions")
        if repetitions < 1:
            raise ValueError("repetitions must be positive")

        warmup_runs = _as_integer(warmup_runs, "warmup_runs")
        if warmup_runs < 0:
            raise ValueError("warmup_runs must be nonnegative")

        self._context = context
        self._kernel = kernel
        self._latency_dtypes = latency_dtypes
        self._input_arguments = input_arguments

        self._pass_count = latency_dtypes["pass_count"].type(_PASS_COUNT)
        self._chain_length = latency_dtypes["chain_length"].type(chain_length)
        self._repetitions = repetitions
        self._warmup_runs = warmup_runs

    def measure(self, batch: inputs.InputBatch) -> LatencyMeasurement:
        snapshot = self._snapshot_batch(batch)
        input_arrays = tuple(
            snapshot.arguments[input_argument.name]
            for input_argument in self._input_arguments
        )

        device_info = self._context.device_info
        elapsed_ticks = np.empty(
            (snapshot.size, self._repetitions),
            dtype=self._latency_dtypes["elapsed_ticks"],
        )
        buffers = self._create_buffers()

        self._warmup(input_arrays, buffers, self._warmup_runs)

        for sample_index in range(snapshot.size):
            self._copy_inputs_to_device(input_arrays, buffers, sample_index)

            for repetition in range(self._repetitions):
                elapsed_ticks[sample_index, repetition] = self._measure_chain(buffers)

        return LatencyMeasurement(
            inputs=snapshot,
            elapsed_ticks=elapsed_ticks,
            device_info=device_info,
            pass_count=int(self._pass_count),
            chain_length=int(self._chain_length),
            warmup_runs=self._warmup_runs,
        )

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
        sample_index: int,
    ) -> None:
        for destination, source in zip(buffers.inputs, input_arrays, strict=True):
            self._context.copy_to_device(
                destination, source[sample_index : sample_index + 1]
            )

    def _run_chain(self, buffers: _LatencyBuffers) -> None:
        self._context.launch(
            self._kernel,
            self._pass_count,
            self._chain_length,
            buffers.mask,
            buffers.elapsed_ticks,
            buffers.output,
            *buffers.inputs,
            config=LaunchConfig(),
        )

    def _warmup(
        self,
        input_arrays: tuple[NDArray[Scalar], ...],
        buffers: _LatencyBuffers,
        runs: int,
    ) -> None:
        for run in range(runs):
            sample_index = run % input_arrays[0].size
            self._copy_inputs_to_device(input_arrays, buffers, sample_index)
            self._run_chain(buffers)

    def _measure_chain(self, buffers: _LatencyBuffers) -> np.uint64:
        self._run_chain(buffers)
        self._context.copy_to_host(buffers.host_elapsed_ticks, buffers.elapsed_ticks)
        return buffers.host_elapsed_ticks[0]


def analyze(
    measurement: LatencyMeasurement,
    *,
    bootstrap_resamples: int,
    confidence_level: float,
    rng: np.random.Generator,
) -> LatencyResult:
    times = measurement.device_info.timer.to_nanoseconds(measurement.elapsed_ticks)
    latencies = np.median(times / measurement.chain_length, axis=1)

    mean_confidence_interval = compute_mean_confidence_interval(
        latencies,
        resamples=bootstrap_resamples,
        confidence_level=confidence_level,
        rng=rng,
    )

    return LatencyResult(
        measurement=measurement,
        latencies=latencies,
        mean=float(np.mean(latencies)),
        mean_confidence_interval=mean_confidence_interval,
        confidence_level=confidence_level,
        bootstrap_resamples=bootstrap_resamples,
        stdev=float(np.std(latencies, ddof=1)) if latencies.size > 1 else None,
        min=float(np.min(latencies)),
        max=float(np.max(latencies)),
    )


def _as_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")

    try:
        return index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
