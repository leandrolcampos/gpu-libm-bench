# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Terminal benchmark reports and experiment provenance."""

from rich.console import Console, Group
from rich.text import Text

from gpu_libm_bench.compiler.images import DeviceImageBuildInfo
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

    metric = result.config.metric
    report = Group(
        Text(f"{metric.name.title()} ({metric.unit})", style="bold"),
        Indented(
            Group(
                Text(_device_label(result.device_info), style="bold"),
                _metric_report(result),
            ),
            top=1,
        ),
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

    console.print(Indented(report, top=1, bottom=1), soft_wrap=True)

    if show_details:
        console.rule(Text("Configuration and Provenance", style="bold"))
        console.print(_configuration_report(result), soft_wrap=True)


def _device_label(device_info: DeviceInfo) -> str:
    return (
        f"{device_info.backend.upper()} — {device_info.id}: "
        f"{device_info.name} ({device_info.architecture})"
    )


def _metric_report(experiment: ExperimentResult) -> Group:
    metric = experiment.config.metric.name
    if metric == "latency":
        return _latency.build_report(experiment)
    raise ValueError(f"unsupported report metric: {metric!r}")


def _configuration_report(experiment: ExperimentResult) -> Group:
    metric = experiment.config.metric.name
    if metric == "latency":
        metric_details = _latency.build_details(experiment.results)
    else:
        raise ValueError(f"unsupported report metric: {metric!r}")

    libraries = []
    for result in experiment.results:
        image_build_info = result.image_build_info
        libraries.append(
            Group(
                Text(f"Library: {image_build_info.image.library}", style="bold"),
                Indented(_image_details(image_build_info), bottom=1),
            )
        )

    return Group(
        Indented(_device_details(experiment.device_info), top=1, bottom=1),
        Indented(
            Group(
                Text(f"Seed: {experiment.config.run.seed}"),
                Text(
                    "Environment: "
                    + "; ".join(
                        f"{name}={value}"
                        for name, value in experiment.environment.items()
                    )
                ),
                Indented(metric_details, left=0, top=1, bottom=1),
                *libraries,
            )
        ),
    )


def _device_details(device_info: DeviceInfo) -> Group:
    timer_info = device_info.timer_info
    return Group(
        Text(_device_label(device_info), style="bold"),
        Text(f"Memory: {device_info.total_memory_bytes / 2**20:,.2f} MiB"),
        Text(
            f"Timer: {timer_info.source}, {timer_info.counter_bits} bits, "
            f"{timer_info.frequency_hz} Hz"
        ),
    )


def _image_details(image_build_info: DeviceImageBuildInfo) -> Group:
    image = image_build_info.image
    target = image.target

    return Group(
        Text(
            f"Compiler: {image_build_info.compiler}\n"
            f"{image_build_info.compiler_version}"
        ),
        Text(
            f"SDK: {image_build_info.sdk_name} {image_build_info.sdk_version}; "
            f"root: {image_build_info.sdk_root}"
        ),
        Text(f"Target: {target.target_triple}; architecture: {target.architecture}"),
        Text(
            f"Bitcode: {image_build_info.bitcode_path}\n"
            f"Device image: {image.image_path}"
        ),
        Text(
            f"Compile options: {' '.join(image_build_info.compile_options) or '(none)'}"
            f"\nLink options: {' '.join(image_build_info.link_options) or '(none)'}"
        ),
    )
