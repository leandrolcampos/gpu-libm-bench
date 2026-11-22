# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Terminal presentation of throughput statistics and launch details."""

from collections import Counter
from collections.abc import Sequence

from rich.console import Group
from rich.text import Text

from gpu_libm_bench.experiment import ExperimentResult, ThroughputResults
from gpu_libm_bench.experiment._throughput import (
    CONFIRMATION_REPETITIONS,
    PLATEAU_MARGIN,
    ThroughputDomainResult,
)

from ._layout import build_domain_report, format_number


def build_report(experiment: ExperimentResult) -> Group:
    libraries = experiment.results_for(ThroughputResults)
    rows = {
        domain: tuple(_format_row(library, domain) for library in libraries)
        for domain in libraries[0].domains
    }
    return build_domain_report(
        (
            "Library",
            "Mean",
            "StDev",
            "Min",
            "Max",
            "Sample size",
            "Coverage",
            "Saturation",
        ),
        rows,
        experiment.generator_descriptions,
    )


def _format_row(library: ThroughputResults, domain: str) -> tuple[str, ...]:
    result = library.domains[domain]
    completed = sum(measurement.complete for measurement in result.search_measurements)
    reported = result.reported_result
    statistics = (
        (
            format_number(reported.mean),
            format_number(reported.stdev),
            format_number(reported.min),
            format_number(reported.max),
            str(reported.measurement.accepted_count),
        )
        if reported is not None
        else ("N/A",) * 5
    )
    return (
        library.image_build_info.image.library,
        *statistics,
        f"{completed}/{len(library.plan.candidates)}",
        _saturation_label(result),
    )


def _saturation_label(result: ThroughputDomainResult) -> str:
    if result.confirmation is None:
        return "No result"
    if result.confirmation.unconfirmed_reason is None:
        return "Confirmed plateau"
    return "Unconfirmed"


def build_details(results: Sequence[ThroughputResults]) -> Group:
    sections = [
        Text(
            "Method: independent calls (input loads and output checksums included)\n"
            "Timer: CUDA events around each kernel launch\n"
            "Estimator: total accepted calls divided by total GPU time\n"
            "Selection: highest rate among complete samples in the search\n"
            f"Confirmation: {CONFIRMATION_REPETITIONS} fresh repetitions, "
            "paired 90% contrast-to-winner rate-ratio intervals within "
            f"±{PLATEAU_MARGIN:.0%}\n"
            "Reported rate: confirmation estimate for a confirmed local plateau, "
            "otherwise the maximum observed in the search"
        )
    ]
    for library in results:
        name = library.image_build_info.image.library
        plan = library.plan
        sections.append(Text(f"\n{name}", style="bold"))
        sections.append(
            Text(
                f"Batch selection: {plan.batch_source}\n"
                f"Search plan: {len(plan.candidates)}/{len(plan.eligible)} "
                f"eligible configurations selected, {len(plan.exclusions)} exclusions"
            )
        )
        for domain, result in library.domains.items():
            sections.append(Text(f"\n{name} · {domain}", style="bold"))
            sections.extend(_domain_details(result))
    return Group(*sections)


def _domain_details(result: ThroughputDomainResult) -> list[Text]:
    complete = sum(measurement.complete for measurement in result.search_measurements)
    details = [
        Text(
            f"Complete search samples: {complete}/"
            f"{len(result.search_measurements)} attempted"
        )
    ]
    if result.reported_result is None:
        details.append(Text("No configuration completed collection."))
        return details

    details.extend(_confirmation_details(result))
    measurement = result.reported_result.measurement
    launch = measurement.launch
    kernel = launch.kernel_info
    discards = Counter(
        attempt.discard_reason
        for attempt in measurement.attempts
        if attempt.discard_reason is not None
    )
    discard_counts = ", ".join(
        f"{reason}: {count}" for reason, count in discards.items()
    )
    clock = (
        f"{measurement.min_clock_hz:,.0f} Hz"
        if measurement.min_clock_hz > 0
        else "disabled"
    )
    details.append(
        Text(
            f"Kernel: {launch.request.symbol}\n"
            f"Unroll factor: {launch.request.unroll_factor}\n"
            f"Batch size: {launch.batch_size:,} calls\n"
            f"Block: {launch.config.block_dim}, grid: {launch.config.grid_dim}\n"
            f"Estimated resident blocks per multiprocessor: "
            f"{launch.max_resident_blocks_per_multiprocessor}\n"
            f"Registers per thread: {kernel.registers_per_thread}\n"
            f"Static shared memory per block: {kernel.static_shared_memory_bytes} B\n"
            f"Local memory per thread: {kernel.local_memory_bytes_per_thread} B\n"
            f"Warmup runs: {measurement.warmup_runs}\n"
            f"Clock threshold: {clock}\n"
            f"Discarded attempts: {discard_counts or '0'}"
        )
    )
    return details


def _confirmation_details(result: ThroughputDomainResult) -> list[Text]:
    confirmation = result.confirmation
    assert result.search_winner is not None and confirmation is not None
    state = (
        confirmation.unconfirmed_reason
        or "Confirmed plateau within tested neighborhood"
    )
    estimate = confirmation.winner_estimate
    repetitions = sum(sample.complete for sample in confirmation.winner_measurements)
    details = [
        Text(
            f"Saturation: {state}\n"
            f"Search winner: {format_number(result.search_winner.mean)} Gcall/s\n"
            f"Winner confirmation: {repetitions}/{CONFIRMATION_REPETITIONS} complete"
            + (
                f", {format_number(estimate.mean)} Gcall/s"
                if estimate is not None
                else ""
            )
        )
    ]
    for contrast in confirmation.contrasts:
        launch = contrast.launch
        if launch is None:
            details.append(Text(f"{contrast.label}: unavailable"))
            continue
        complete = sum(sample.complete for sample in contrast.measurements)
        interval = contrast.ratio_interval
        evidence = (
            f"90% ratio interval [{interval.lower:.3f}, {interval.upper:.3f}]"
            if interval is not None
            else "ratio interval unavailable"
        )
        details.append(
            Text(
                f"{contrast.label}: {launch.batch_size:,} calls, "
                f"block {launch.config.block_dim[0]}, "
                f"grid {launch.config.grid_dim[0]}, "
                f"{complete}/{CONFIRMATION_REPETITIONS} complete, {evidence}"
            )
        )
    return details
