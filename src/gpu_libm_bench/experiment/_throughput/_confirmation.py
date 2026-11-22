# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Fixed-budget confirmation of the throughput winner and its neighbors."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from math import exp, log, sqrt
from statistics import mean, stdev

from gpu_libm_bench.benchmarks.throughput import (
    ThroughputMeasurement,
    ThroughputPlannedLaunch,
    ThroughputResult,
    analyze,
)
from gpu_libm_bench.inputs import InputGenerator

CONFIRMATION_REPETITIONS = 5
PLATEAU_MARGIN = 0.05
# The 0.95 Student t quantile for four degrees of freedom.
_T_CRITICAL = 2.131846786


@dataclass(frozen=True, slots=True)
class RateRatioInterval:
    lower: float
    upper: float


@dataclass(frozen=True, slots=True)
class ThroughputContrast:
    label: str
    launch: ThroughputPlannedLaunch | None
    measurements: tuple[ThroughputMeasurement, ...]
    ratio_interval: RateRatioInterval | None


@dataclass(frozen=True, slots=True)
class ThroughputConfirmation:
    winner_measurements: tuple[ThroughputMeasurement, ...]
    contrasts: tuple[ThroughputContrast, ...]
    winner_estimate: ThroughputResult | None
    unconfirmed_reason: str | None


def confirm_winner(
    winner: ThroughputResult,
    eligible: Sequence[ThroughputPlannedLaunch],
    generators: Sequence[InputGenerator],
    measure: Callable[[ThroughputPlannedLaunch, InputGenerator], ThroughputMeasurement],
    *,
    search_complete: bool,
) -> ThroughputConfirmation:
    if len(generators) != CONFIRMATION_REPETITIONS:
        raise ValueError("confirmation requires one generator per repetition")

    winner_launch = winner.measurement.launch
    neighbors = _select_contrasts(winner_launch, eligible)
    launches = (
        winner_launch,
        *(launch for _, launch in neighbors if launch is not None),
    )
    series: dict[ThroughputPlannedLaunch, list[ThroughputMeasurement]] = {
        launch: [] for launch in launches
    }
    for repetition, generator in enumerate(generators):
        # Move each configuration through the execution order across repetitions.
        offset = repetition % len(launches)
        for launch in launches[offset:] + launches[:offset]:
            series[launch].append(measure(launch, generator))

    winner_measurements = tuple(series[winner_launch])
    winner_rates = _get_complete_rates(winner_measurements)
    contrasts = []
    for label, neighbor in neighbors:
        measurements = tuple(series[neighbor]) if neighbor is not None else ()
        rates = _get_complete_rates(measurements)
        contrasts.append(
            ThroughputContrast(
                label=label,
                launch=neighbor,
                measurements=measurements,
                ratio_interval=(
                    _get_paired_ratio_interval(winner_rates, rates)
                    if winner_rates is not None and rates is not None
                    else None
                ),
            )
        )

    estimate = (
        _combine_winner_samples(winner_measurements)
        if winner_rates is not None
        else None
    )
    return ThroughputConfirmation(
        winner_measurements=winner_measurements,
        contrasts=tuple(contrasts),
        winner_estimate=estimate,
        unconfirmed_reason=_get_unconfirmed_reason(
            winner, estimate, contrasts, search_complete
        ),
    )


def _select_contrasts(
    winner: ThroughputPlannedLaunch, eligible: Sequence[ThroughputPlannedLaunch]
) -> tuple[tuple[str, ThroughputPlannedLaunch | None], ...]:
    block = winner.config.block_dim[0]
    grid = winner.config.grid_dim[0]
    larger_batch = None
    other_grids = []
    other_blocks = []

    for launch in eligible:
        if launch.request != winner.request or launch == winner:
            continue
        if launch.batch_size > winner.batch_size and launch.config == winner.config:
            if larger_batch is None or launch.batch_size < larger_batch.batch_size:
                larger_batch = launch
        elif launch.batch_size == winner.batch_size:
            if launch.config.block_dim[0] == block:
                other_grids.append(launch)
            else:
                other_blocks.append(launch)

    other_grid = min(
        other_grids,
        key=lambda launch: (
            launch.config.grid_dim[0] < grid,
            abs(launch.config.grid_dim[0] - grid),
        ),
        default=None,
    )
    winner_grid_to_residency = Fraction(
        grid, winner.max_resident_blocks_per_multiprocessor
    )
    other_block = min(
        other_blocks,
        key=lambda launch: abs(
            Fraction(
                launch.config.grid_dim[0], launch.max_resident_blocks_per_multiprocessor
            )
            - winner_grid_to_residency
        ),
        default=None,
    )
    return (
        ("Larger batch", larger_batch),
        ("Grid", other_grid),
        ("Block size", other_block),
    )


def _get_complete_rates(
    measurements: tuple[ThroughputMeasurement, ...],
) -> tuple[float, ...] | None:
    if len(measurements) != CONFIRMATION_REPETITIONS or not all(
        measurement.complete for measurement in measurements
    ):
        return None
    rates = []
    for measurement in measurements:
        result = analyze(measurement)
        assert result is not None
        rates.append(result.mean)
    return tuple(rates)


def _get_paired_ratio_interval(
    winner_rates: tuple[float, ...], contrast_rates: tuple[float, ...]
) -> RateRatioInterval:
    log_ratios = tuple(
        log(contrast) - log(winner)
        for winner, contrast in zip(winner_rates, contrast_rates, strict=True)
    )
    center = mean(log_ratios)
    half_width = _T_CRITICAL * stdev(log_ratios) / sqrt(len(log_ratios))
    return RateRatioInterval(exp(center - half_width), exp(center + half_width))


def _combine_winner_samples(
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


def _get_unconfirmed_reason(
    winner: ThroughputResult,
    estimate: ThroughputResult | None,
    contrasts: Sequence[ThroughputContrast],
    search_complete: bool,
) -> str | None:
    if estimate is None:
        return "Incomplete winner confirmation"

    lower_limit = 1 - PLATEAU_MARGIN
    upper_limit = 1 + PLATEAU_MARGIN
    for contrast in contrasts:
        if (
            contrast.ratio_interval is not None
            and contrast.ratio_interval.lower > upper_limit
        ):
            return f"Gain beyond the equivalence margin: {contrast.label.lower()}"

    if not search_complete:
        return "Incomplete search collection"
    for contrast in contrasts:
        if contrast.launch is None:
            return f"No viable {contrast.label.lower()} contrast"
        if contrast.ratio_interval is None:
            return f"Incomplete {contrast.label.lower()} contrast collection"
        if (
            contrast.ratio_interval.lower < lower_limit
            or contrast.ratio_interval.upper > upper_limit
        ):
            return f"Equivalence not established: {contrast.label.lower()}"

    if not lower_limit <= estimate.mean / winner.mean <= upper_limit:
        return "Winner rate changed between search and confirmation"
    return None
