# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Terminal presentation of throughput statistics and launch details."""

from collections import Counter
from collections.abc import Sequence

from rich.console import Group
from rich.text import Text

from gpu_libm_bench.experiment import ExperimentResult, ThroughputResults
from gpu_libm_bench.experiment._throughput import ThroughputDomainResult

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
    completed = sum(measurement.complete for measurement in result.measurements)
    total = len(result.measurements) + len(library.exclusions)
    best = result.best
    statistics = (
        (
            format_number(best.mean),
            format_number(best.stdev),
            format_number(best.min),
            format_number(best.max),
            str(best.measurement.accepted_count),
        )
        if best is not None
        else ("N/A",) * 5
    )
    return (
        library.image_build_info.image.library,
        *statistics,
        f"{completed}/{total}",
        "Not evaluated",
    )


def build_details(results: Sequence[ThroughputResults]) -> Group:
    sections = [
        Text(
            "Method: independent calls (input loads and output checksums included)\n"
            "Timer: CUDA events around each kernel launch\n"
            "Estimator: total accepted calls divided by total GPU time\n"
            "Selection: highest mean among configurations with a complete sample\n"
            "Saturation: not evaluated (one batch size and launch geometry per variant)"
        )
    ]
    for library in results:
        name = library.image_build_info.image.library
        for domain, result in library.domains.items():
            sections.append(Text(f"\n{name} · {domain}", style="bold"))
            sections.extend(_domain_details(result))
        if library.exclusions:
            sections.append(Text(f"\nExcluded configurations for {name}", style="bold"))
            for exclusion in library.exclusions:
                sections.append(Text(f"{exclusion.request.symbol}: {exclusion.reason}"))
    return Group(*sections)


def _domain_details(result: ThroughputDomainResult) -> list[Text]:
    details = []
    complete = sum(measurement.complete for measurement in result.measurements)
    details.append(
        Text(f"Complete samples: {complete}/{len(result.measurements)} attempted")
    )
    for measurement in result.measurements:
        if measurement.timed_out:
            accepted = measurement.accepted_count
            discarded = len(measurement.attempts) - accepted
            details.append(
                Text(
                    f"{measurement.launch.request.symbol}: timed out "
                    f"({accepted}/{measurement.sample_size} accepted, "
                    f"{discarded} discarded)"
                )
            )

    if result.best is None:
        details.append(Text("No configuration completed collection."))
        return details

    measurement = result.best.measurement
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
            f"{launch.blocks_per_multiprocessor}\n"
            f"Registers per thread: {kernel.registers_per_thread}\n"
            f"Static shared memory per block: {kernel.static_shared_memory_bytes} B\n"
            f"Local memory per thread: {kernel.local_memory_bytes_per_thread} B\n"
            f"Warmup runs: {measurement.warmup_runs}\n"
            f"Clock threshold: {clock}\n"
            f"Discarded attempts: {discard_counts or '0'}"
        )
    )
    return details
