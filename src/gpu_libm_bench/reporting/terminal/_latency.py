# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Terminal presentation of latency statistics and methodology."""

from collections.abc import Mapping, Sequence

from rich import box
from rich.cells import cell_len
from rich.console import Group
from rich.table import Table
from rich.text import Text

from gpu_libm_bench.benchmarks.latency import LatencyResult
from gpu_libm_bench.experiment import ExperimentResult, LatencyResults

from ._layout import Indented


def build_report(experiment: ExperimentResult) -> Group:
    domain_rows = {
        name: [
            (library.image_build_info.image.library, library[name])
            for library in experiment.results
        ]
        for name in experiment.results[0]
    }

    sections = [
        Indented(
            Group(
                Text(f"Domain: {name}", style="bold"),
                Text(experiment.generator_descriptions[name]),
                table,
            ),
            top=1,
        )
        for name, table in _build_tables(domain_rows).items()
    ]

    return Group(*sections)


def _build_tables(
    domains: Mapping[str, Sequence[tuple[str, LatencyResult]]],
) -> dict[str, Table]:
    formatted = {name: _format_table(rows) for name, rows in domains.items()}
    if not formatted:
        return {}

    headers = [header for header, _ in formatted.values()]
    cells = [row for header, rows in formatted.values() for row in (header, *rows)]
    column_widths = tuple(
        max(cell_len(row[index]) for row in cells) for index in range(len(headers[0]))
    )

    tables = {}
    for name, (header, rows) in formatted.items():
        table = Table(
            box=box.SQUARE,
            width=sum(column_widths) + 3 * len(header) + 1,
        )

        for index, (label, width) in enumerate(zip(header, column_widths, strict=True)):
            table.add_column(
                Text(label, justify="center"),
                width=width,
                justify="left" if index == 0 else "right",
                no_wrap=True,
                overflow="fold",
            )

        for row in rows:
            table.add_row(*(Text(value) for value in row))

        tables[name] = table

    return tables


def _format_table(
    rows: Sequence[tuple[str, LatencyResult]],
) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
    cells = []
    for library, result in rows:
        cells.append(
            (
                library,
                _format_number(result.mean),
                _format_number(result.stdev) if result.stdev is not None else "N/A",
                _format_number(result.min),
                _format_number(result.max),
                str(result.measurement.inputs.size),
            )
        )

    return (
        ("Library", "Mean", "StDev", "Min", "Max", "Samples"),
        tuple(cells),
    )


def _format_number(value: float) -> str:
    return f"{value:.5g}"


def build_details(results: Sequence[LatencyResults]) -> Group:
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

    sections = []
    for (method, libraries), domains in groups.items():
        section = Group(Text(f"Domains: {', '.join(domains)}"), Text(method))
        if len(libraries) != len(results):
            section = Group(Text(f"Libraries: {', '.join(libraries)}"), section)
        sections.append(section)

    return Group(*sections)


def _methodology(result: LatencyResult) -> Text:
    measurement = result.measurement
    repetitions = measurement.elapsed_ticks.shape[1]
    return Text(
        "Method: dependent-call chain (overhead included)\n"
        "Estimator: mean of per-sample medians\n"
        f"Repetitions: {repetitions}\n"
        f"Chain length: {measurement.chain_length}\n"
        f"Passes: {measurement.pass_count}\n"
        f"Warmup runs: {measurement.warmup_runs}"
    )
