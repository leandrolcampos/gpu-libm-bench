# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Terminal presentation of throughput statistics and methodology."""

from collections import Counter
from collections.abc import Sequence

from rich.console import Group
from rich.text import Text

from gpu_libm_bench.configs import ThroughputConfig
from gpu_libm_bench.experiment import ExperimentResult, ThroughputResults
from gpu_libm_bench.experiment._throughput import ThroughputDomainResult

from ._layout import build_domain_report, format_number


def build_report(experiment: ExperimentResult) -> Group:
    libraries = experiment.results_for(ThroughputResults)
    rows = {
        domain: tuple(
            _format_row(
                library.image_build_info.image.library,
                library[domain],
                candidate_count=len(library.plan.candidates),
            )
            for library in libraries
        )
        for domain in libraries[0]
    }
    return build_domain_report(
        (
            "Library",
            "Aggregate",
            "StDev",
            "Min",
            "Max",
            "Launches",
            "Candidates",
        ),
        rows,
        experiment.generator_descriptions,
    )


def _format_row(
    library: str, result: ThroughputDomainResult, *, candidate_count: int
) -> tuple[str, ...]:
    completed = sum(measurement.complete for measurement in result.search_measurements)
    confirmation = result.confirmation
    reported = result.reported_result
    if confirmation is not None and reported is not None:
        measurements = confirmation.reference_measurements
        statistics = (
            format_number(reported.aggregate),
            format_number(reported.stdev),
            format_number(reported.min),
            format_number(reported.max),
            f"{measurements[0].accepted_count} × {len(measurements)}",
        )
    else:
        statistics = ("N/A",) * 5
    return library, *statistics, f"{completed}/{candidate_count}"


def build_details(
    results: Sequence[ThroughputResults], metric: ThroughputConfig
) -> Group:
    sections = []
    for library in results:
        name = library.image_build_info.image.library
        plan = library.plan
        domains = [
            Group(
                Text(f"\nDomain: {domain}", style="bold"),
                *_domain_details(
                    result, confidence_level=metric.confirmation.confidence_level
                ),
            )
            for domain, result in library.items()
        ]
        sections.append(
            Group(
                Text(f"\nLibrary: {name}", style="bold"),
                Text(
                    f"Batch selection: {plan.batch_source}\n"
                    f"Search plan: {len(plan.candidates)}/{len(plan.eligible)} "
                    f"eligible configurations selected, "
                    f"{len(plan.exclusions)} exclusions"
                ),
                *domains,
            )
        )
    return Group(_methodology(metric), *sections)


def _methodology(metric: ThroughputConfig) -> Text:
    search = metric.search
    confirmation = metric.confirmation
    confidence_percent = confirmation.confidence_level * 100
    gain_threshold_percent = confirmation.relative_gain_threshold * 100
    batch_growth_percent = confirmation.batch_volume_growth_ratio * 100
    contrasts = "larger batch, larger grid, higher unroll factor"
    if metric.pass_count > 1:
        contrasts += ", larger block"
    return Text(
        "Method: independent calls (input loads and output checksums included)\n"
        "Estimator: total accepted calls divided by total GPU time\n"
        "Timer: CUDA events around each kernel launch\n"
        f"Launches: {metric.sample.size} accepted launches per repetition "
        "(sample.size) "
        f"× {confirmation.repetitions} repetitions (confirmation.repetitions).\n"
        f"Passes: {metric.pass_count}\n"
        f"Warmup runs: {metric.warmup_runs}\n"
        "Selection: highest rate among complete samples in the search\n"
        f"Search candidate budget: {search.max_candidates}\n"
        f"Batch volume-to-L2 ratios: {search.batch_volume_to_l2_ratios}\n"
        f"Fallback batch volumes (MiB): {search.fallback_batch_volumes_mib}\n"
        f"Warps per block: {search.warps_per_block}\n"
        f"Waves per grid: {search.waves_per_grid}\n"
        f"Confirmation: one round with {confirmation.repetitions} fresh "
        f"repetitions and one-sided {confidence_percent:.6g}% bounds "
        "for paired rate ratios\n"
        f"Relevant gain threshold: {gain_threshold_percent:.6g}%\n"
        f"Batch volume growth: {batch_growth_percent:.6g}%\n"
        f"Maximum waves per contrast grid: {confirmation.max_waves_per_grid}\n"
        f"Contrasts: {contrasts}\n"
        "Reported rate: fresh estimate at the search winner"
    )


def _domain_details(
    result: ThroughputDomainResult, *, confidence_level: float
) -> list[Text]:
    complete = sum(measurement.complete for measurement in result.search_measurements)
    details = [
        Text(
            f"Complete search samples: {complete}/"
            f"{len(result.search_measurements)} attempted"
        )
    ]
    if result.search_winner is None:
        details.append(Text("Local stability: no configuration completed collection."))
        return details

    details.extend(_confirmation_details(result, confidence_level=confidence_level))
    confirmation = result.confirmation
    assert confirmation is not None
    measurement = (
        confirmation.reference_estimate.measurement
        if confirmation.reference_estimate is not None
        else confirmation.reference_measurements[0]
    )
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
            f"Batch size: {launch.batch_size:,} input tuples\n"
            f"Evaluations per launch: {measurement.evaluations_per_launch:,}\n"
            f"Block: {launch.config.block_dim}, grid: {launch.config.grid_dim}\n"
            f"Estimated resident blocks per multiprocessor: "
            f"{launch.max_resident_blocks_per_multiprocessor}\n"
            f"Registers per thread: {kernel.registers_per_thread}\n"
            f"Static shared memory per block: {kernel.static_shared_memory_bytes} B\n"
            f"Local memory per thread: {kernel.local_memory_bytes_per_thread} B\n"
            f"Clock threshold: {clock}\n"
            f"Discarded attempts: {discard_counts or '0'}"
        )
    )
    return details


def _confirmation_details(
    result: ThroughputDomainResult, *, confidence_level: float
) -> list[Text]:
    confirmation = result.confirmation
    assert result.search_winner is not None and confirmation is not None
    if confirmation.best_gain is not None:
        evidence = f"Relevant gain detected: {confirmation.best_gain.label.lower()}"
    else:
        evidence = (
            confirmation.inconclusive_reason
            or "Relevant gains ruled out in tested contrasts"
        )
    estimate = confirmation.reference_estimate
    repetitions = len(confirmation.reference_measurements)
    complete_reference = sum(
        sample.complete for sample in confirmation.reference_measurements
    )
    details = [
        Text(
            f"Search winner: {format_number(result.search_winner.aggregate)} Gcall/s\n"
            f"Local stability: {evidence}\n"
            f"Reference: {complete_reference}/{repetitions} complete"
            + (
                f", {format_number(estimate.aggregate)} Gcall/s"
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
        bounds = contrast.ratio_bounds
        evidence = (
            f"ratio {bounds.estimate:.3f}, "
            f"{confidence_level * 100:.6g}% lower {bounds.lower:.3f}, "
            f"upper {bounds.upper:.3f}"
            if bounds is not None
            else "ratio bounds unavailable"
        )
        details.append(
            Text(
                f"{contrast.label}: {launch.batch_size:,} input tuples, "
                f"block {launch.config.block_dim[0]}, "
                f"grid {launch.config.grid_dim[0]}, "
                f"unroll {launch.request.unroll_factor}, "
                f"{complete}/{repetitions} complete, {evidence}"
            )
        )
    return details
