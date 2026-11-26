# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Terminal presentation of latency statistics and methodology."""

from collections.abc import Sequence

from rich.console import Group
from rich.text import Text

from gpu_libm_bench.benchmarks.latency import LatencyResult
from gpu_libm_bench.experiment import ExperimentResult, LatencyResults

from ._layout import build_domain_report, format_number


def build_report(experiment: ExperimentResult) -> Group:
    libraries = experiment.results_for(LatencyResults)
    rows = {
        domain: tuple(
            _format_row(library.image_build_info.image.library, library[domain])
            for library in libraries
        )
        for domain in libraries[0]
    }
    return build_domain_report(
        ("Library", "Mean", "StDev", "Min", "Max", "Sample size"),
        rows,
        experiment.generator_descriptions,
    )


def _format_row(library: str, result: LatencyResult) -> tuple[str, ...]:
    return (
        library,
        format_number(result.mean),
        format_number(result.stdev),
        format_number(result.min),
        format_number(result.max),
        str(result.latencies.size),
    )


def build_details(results: Sequence[LatencyResults], return_type: str) -> Group:
    methods: dict[tuple[str, str], list[str]] = {}
    for domain in results[0]:
        for library in results:
            method = _methodology(library[domain]).plain
            methods.setdefault((domain, method), []).append(
                library.image_build_info.image.library
            )

    groups: dict[tuple[str, tuple[str, ...]], list[str]] = {}
    for (domain, method), library_names in methods.items():
        groups.setdefault((method, tuple(library_names)), []).append(domain)

    sections: list[Group | Text] = []
    for (method, libraries), domains in groups.items():
        section = Group(Text(f"Domains: {', '.join(domains)}"), Text(method))
        if len(libraries) != len(results):
            section = Group(Text(f"Libraries: {', '.join(libraries)}"), section)
        sections.append(section)

    estimate = results[0].chain_cost
    if estimate is not None:
        lengths = ", ".join(map(str, estimate.lengths))
        details = f"{return_type}, lengths {lengths}"
        if estimate.cycles_per_call is None:
            cost = f"inconclusive: {estimate.reason}"
        else:
            cost = f"{format_number(estimate.cycles_per_call)} cycles/call"
            details += ", correction pending validation"
        sections.append(Text(f"Estimated dependency chain cost: {cost} ({details})"))

    return Group(*sections)


def _methodology(result: LatencyResult) -> Text:
    measurement = result.measurement
    return Text(
        "Method: dependent-call chain\n"
        "Estimator: mean of per-input latencies\n"
        f"Chain length: {measurement.chain_length}\n"
        f"Passes: {measurement.pass_count}\n"
        f"Warmup runs: {measurement.warmup_runs}"
    )
