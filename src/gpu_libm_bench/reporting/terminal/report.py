# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Terminal benchmark reports and experiment provenance."""

from rich.console import Console, Group
from rich.text import Text

from gpu_libm_bench.devices import DeviceInfo
from gpu_libm_bench.experiment import ExperimentResult

from . import _latency
from ._layout import Indented


def print_result(
    result: ExperimentResult,
    *,
    show_details: bool = False,
    console: Console | None = None,
) -> None:
    console = console if console is not None else Console()
    if not result.results:
        console.print("No benchmark results.")
        return

    report = Group(
        Text(_device_label(result.device_info), style="bold"),
        Indented(_metric_report(result), top=1),
    )

    function = result.config.function
    console.rule(Text(function.name, style="bold"))
    arguments = ", ".join(
        f"{argument.type} {argument.name}" for argument in function.signature.arguments
    )
    console.print(
        Text(f"{function.signature.return_type} {function.name}({arguments})"),
        soft_wrap=True,
    )

    console.print(Indented(report, bottom=1), soft_wrap=True)

    if show_details:
        console.rule(Text("Configuration and Provenance", style="bold"))
        console.print(_configuration_report(result), soft_wrap=True)


def _device_label(info: DeviceInfo) -> str:
    return (
        f"{info.backend.upper()} — {info.device_id}: {info.name} ({info.architecture})"
    )


def _metric_report(experiment: ExperimentResult) -> Group:
    metric = experiment.config.metric.name
    if metric == "latency":
        return _latency.build_report(experiment)
    raise ValueError(f"unsupported report metric: {metric!r}")


def _configuration_report(experiment: ExperimentResult) -> Group:
    metric = experiment.config.metric.name
    if metric == "latency":
        metric_details = _latency.build_details(experiment)
    else:
        raise ValueError(f"unsupported report metric: {metric!r}")

    return Group(
        Indented(_device_details(experiment.device_info), top=1, bottom=1),
        Indented(
            Group(
                Text(f"Seed: {experiment.config.run.seed}"),
                Text(f"Library: {experiment.image_build_info.image.library}"),
                metric_details,
                _experiment_details(experiment),
            )
        ),
    )


def _device_details(info: DeviceInfo) -> Group:
    timer = info.timer
    return Group(
        Text(_device_label(info), style="bold"),
        Text(f"Memory: {info.total_memory_bytes / 2**20:,.2f} MiB"),
        Text(
            f"Timer: {timer.source}, {timer.counter_bits} bits, {timer.frequency_hz} Hz"
        ),
    )


def _experiment_details(experiment: ExperimentResult) -> Group:
    build_info = experiment.image_build_info
    image = build_info.image
    target = image.target

    return Group(
        Text(
            "Environment: "
            + "; ".join(
                f"{name}={value}" for name, value in experiment.environment.items()
            )
        ),
        Text(f"Compiler: {build_info.compiler}\n{build_info.compiler_version}"),
        Text(
            f"SDK: {build_info.sdk_name} {build_info.sdk_version}; root: {build_info.sdk_root}"
        ),
        Text(f"Target: {target.target_triple}; architecture: {target.architecture}"),
        Text(f"Bitcode: {build_info.bitcode_file}\nImage: {image.path}"),
        Text(
            f"Compile options: {' '.join(build_info.compile_options) or '(none)'}\n"
            f"Link options: {' '.join(build_info.link_options) or '(none)'}"
        ),
    )
