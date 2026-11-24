# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Confirmation of the selected throughput launch configuration."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from math import exp, isfinite, log, sqrt
from statistics import mean, stdev

from scipy.stats import t  # type: ignore[import-untyped]

from gpu_libm_bench.benchmarks.throughput import (
    ThroughputMeasurement,
    ThroughputPlannedLaunch,
    ThroughputResult,
    analyze,
    try_increase_throughput_batch,
    try_plan_throughput_launch,
)
from gpu_libm_bench.configs import ThroughputConfig
from gpu_libm_bench.devices import DeviceInfo, DeviceKernel
from gpu_libm_bench.inputs import InputGenerator
from gpu_libm_bench.kernels import KernelRequest


@dataclass(frozen=True, slots=True)
class _RateRatioBounds:
    estimate: float
    lower: float
    upper: float


@dataclass(frozen=True, slots=True)
class _Contrast:
    label: str
    launch: ThroughputPlannedLaunch | None
    measurements: tuple[ThroughputMeasurement, ...]
    ratio_bounds: _RateRatioBounds | None


@dataclass(frozen=True, slots=True)
class ThroughputConfirmation:
    reference_measurements: tuple[ThroughputMeasurement, ...]
    reference_estimate: ThroughputResult | None
    contrasts: tuple[_Contrast, ...]
    best_gain: _Contrast | None
    inconclusive_reason: str | None


def confirm_winner(
    winner: ThroughputPlannedLaunch,
    eligible: Sequence[ThroughputPlannedLaunch],
    device_info: DeviceInfo,
    kernels: Mapping[KernelRequest, DeviceKernel],
    metric: ThroughputConfig,
    generators: Sequence[InputGenerator],
    measure: Callable[[ThroughputPlannedLaunch, InputGenerator], ThroughputMeasurement],
) -> ThroughputConfirmation:
    confirmation = metric.confirmation
    if len(generators) != confirmation.repetitions:
        raise ValueError("confirmation requires one input generator per repetition")
    confidence_level = confirmation.confidence_level
    if not 0.5 < confidence_level < 1:
        raise ValueError(
            "metric.confirmation.confidence_level must be "
            "greater than 0.5 and less than 1"
        )
    relative_gain_threshold = confirmation.relative_gain_threshold
    if not isfinite(relative_gain_threshold) or relative_gain_threshold < 0:
        raise ValueError(
            "metric.confirmation.relative_gain_threshold must be finite and nonnegative"
        )

    critical_value = float(t.ppf(confidence_level, df=confirmation.repetitions - 1))
    neighbors = _plan_contrasts(
        winner,
        eligible,
        device_info,
        kernels,
        include_larger_block=metric.pass_count > 1,
        batch_volume_growth_ratio=confirmation.batch_volume_growth_ratio,
        max_waves_per_grid=confirmation.max_waves_per_grid,
    )
    return _measure_confirmation(
        winner,
        neighbors,
        generators,
        measure,
        critical_value=critical_value,
        gain_ratio=1 + relative_gain_threshold,
    )


def _plan_contrasts(
    reference: ThroughputPlannedLaunch,
    eligible: Sequence[ThroughputPlannedLaunch],
    device_info: DeviceInfo,
    kernels: Mapping[KernelRequest, DeviceKernel],
    *,
    include_larger_block: bool,
    batch_volume_growth_ratio: float,
    max_waves_per_grid: int,
) -> tuple[tuple[str, ThroughputPlannedLaunch | None], ...]:
    if max_waves_per_grid < 1:
        raise ValueError("metric.confirmation.max_waves_per_grid must be at least 1")

    request = reference.request
    unroll_factor = request.unroll_factor
    assert unroll_factor is not None
    batch_size = reference.batch_size
    threads_per_block = reference.config.block_dim[0]
    blocks_in_grid = reference.config.grid_dim[0]

    kernel = kernels[request]
    batch_contrast = try_increase_throughput_batch(
        reference,
        device_info,
        kernel,
        batch_volume_growth_ratio=batch_volume_growth_ratio,
    )

    blocks_per_wave = (
        device_info.multiprocessor_count
        * reference.max_resident_blocks_per_multiprocessor
    )
    next_grid_size = min(
        2 * blocks_in_grid,
        max_waves_per_grid * blocks_per_wave,
        batch_size // (threads_per_block * unroll_factor),
        device_info.launch_limits.max_grid_dim[0],
    )
    grid_contrast = None
    if next_grid_size > blocks_in_grid:
        grid_contrast = try_plan_throughput_launch(
            device_info,
            kernel,
            request,
            batch_size=batch_size,
            threads_per_block=threads_per_block,
            blocks_in_grid=next_grid_size,
        )

    next_request = next(
        (
            candidate
            for candidate in kernels
            if candidate.unroll_factor == 2 * unroll_factor
        ),
        None,
    )
    unroll_contrast = None
    if next_request is not None:
        unroll_contrast = try_plan_throughput_launch(
            device_info,
            kernels[next_request],
            next_request,
            batch_size=batch_size,
            threads_per_block=threads_per_block,
            blocks_in_grid=blocks_in_grid,
        )

    contrasts = (
        ("Larger batch", batch_contrast),
        ("Larger grid", grid_contrast),
        ("Higher unroll factor", unroll_contrast),
    )
    if not include_larger_block:
        return contrasts
    return (
        *contrasts,
        ("Larger block", _plan_larger_block(reference, eligible, device_info, kernel)),
    )


def _plan_larger_block(
    reference: ThroughputPlannedLaunch,
    eligible: Sequence[ThroughputPlannedLaunch],
    device_info: DeviceInfo,
    kernel: DeviceKernel,
) -> ThroughputPlannedLaunch | None:
    larger_blocks = sorted(
        {
            launch.config.block_dim[0]
            for launch in eligible
            if launch.request == reference.request
            and launch.config.block_dim[0] > reference.config.block_dim[0]
        }
    )
    for threads_per_block in larger_blocks:
        block_contrast = try_plan_throughput_launch(
            device_info,
            kernel,
            reference.request,
            batch_size=reference.batch_size,
            threads_per_block=threads_per_block,
            blocks_in_grid=reference.config.grid_dim[0],
        )
        if block_contrast is not None:
            return block_contrast
    return None


def _measure_confirmation(
    reference: ThroughputPlannedLaunch,
    neighbors: Sequence[tuple[str, ThroughputPlannedLaunch | None]],
    generators: Sequence[InputGenerator],
    measure: Callable[[ThroughputPlannedLaunch, InputGenerator], ThroughputMeasurement],
    *,
    critical_value: float,
    gain_ratio: float,
) -> ThroughputConfirmation:
    launches = (reference, *(launch for _, launch in neighbors if launch is not None))
    series: dict[ThroughputPlannedLaunch, list[ThroughputMeasurement]] = {
        launch: [] for launch in launches
    }
    for repetition, generator in enumerate(generators):
        # Move each configuration through the execution order across repetitions.
        offset = repetition % len(launches)
        for launch in launches[offset:] + launches[:offset]:
            series[launch].append(measure(launch, generator))

    reference_measurements = tuple(series[reference])
    reference_rates = _get_complete_rates(reference_measurements)
    reference_estimate = (
        _estimate_reference(reference_measurements)
        if reference_rates is not None
        else None
    )

    contrasts = []
    best_gain = None
    best_gain_ratio = gain_ratio
    for label, neighbor in neighbors:
        measurements = tuple(series[neighbor]) if neighbor is not None else ()
        rates = _get_complete_rates(measurements)
        bounds = (
            _compute_paired_ratio_bounds(
                reference_rates, rates, critical_value=critical_value
            )
            if reference_rates is not None and rates is not None
            else None
        )
        contrast = _Contrast(label, neighbor, measurements, bounds)
        contrasts.append(contrast)
        if (
            bounds is not None
            and bounds.lower > gain_ratio
            and bounds.estimate > best_gain_ratio
        ):
            best_gain = contrast
            best_gain_ratio = bounds.estimate

    return ThroughputConfirmation(
        reference_measurements=reference_measurements,
        reference_estimate=reference_estimate,
        contrasts=tuple(contrasts),
        best_gain=best_gain,
        inconclusive_reason=(
            _get_inconclusive_reason(
                reference_estimate, contrasts, gain_ratio=gain_ratio
            )
            if best_gain is None
            else None
        ),
    )


def _get_inconclusive_reason(
    reference: ThroughputResult | None,
    contrasts: Sequence[_Contrast],
    *,
    gain_ratio: float,
) -> str | None:
    if reference is None:
        return "Incomplete reference collection"
    batch_contrast, *parallelism_contrasts = contrasts
    if batch_contrast.launch is None:
        return "No viable larger batch contrast"
    if all(contrast.launch is None for contrast in parallelism_contrasts):
        return "No viable parallelism contrast"

    for contrast in contrasts:
        if contrast.launch is None:
            continue
        if contrast.ratio_bounds is None:
            return f"Incomplete {contrast.label.lower()} contrast collection"
        if contrast.ratio_bounds.upper > gain_ratio:
            return f"Relevant gain not ruled out: {contrast.label.lower()}"
    return None


def _get_complete_rates(
    measurements: tuple[ThroughputMeasurement, ...],
) -> tuple[float, ...] | None:
    if not measurements:
        return None

    rates = []
    for measurement in measurements:
        result = analyze(measurement)
        if result is None:
            return None
        rates.append(result.aggregate)
    return tuple(rates)


def _compute_paired_ratio_bounds(
    reference_rates: tuple[float, ...],
    contrast_rates: tuple[float, ...],
    *,
    critical_value: float,
) -> _RateRatioBounds:
    log_ratios = tuple(
        log(contrast) - log(reference)
        for reference, contrast in zip(reference_rates, contrast_rates, strict=True)
    )
    center = mean(log_ratios)
    half_width = critical_value * stdev(log_ratios) / sqrt(len(log_ratios))
    return _RateRatioBounds(
        exp(center), exp(center - half_width), exp(center + half_width)
    )


def _estimate_reference(
    measurements: tuple[ThroughputMeasurement, ...],
) -> ThroughputResult:
    first = measurements[0]
    # The aggregate rate retains the usual calls-over-total-GPU-time estimator.
    combined = replace(
        first,
        sample_size=len(measurements) * first.sample_size,
        warmup_runs=sum(measurement.warmup_runs for measurement in measurements),
        attempts=tuple(
            replace(
                attempt,
                batch_index=attempt.batch_index + repetition * first.sample_size,
            )
            for repetition, measurement in enumerate(measurements)
            for attempt in measurement.attempts
        ),
    )
    result = analyze(combined)
    assert result is not None
    return result
