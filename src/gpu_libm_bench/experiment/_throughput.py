# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Throughput execution and result aggregation for experiments."""

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

from gpu_libm_bench.benchmarks.throughput import (
    ThroughputBenchmark,
    ThroughputExcludedLaunch,
    ThroughputMeasurement,
    ThroughputResult,
    analyze,
    try_plan_throughput_launch,
)
from gpu_libm_bench.compiler.images import DeviceImageBuildInfo
from gpu_libm_bench.configs import FunctionConfig, ThroughputConfig
from gpu_libm_bench.devices import DeviceContext, DeviceInfo
from gpu_libm_bench.inputs import InputGenerator
from gpu_libm_bench.kernels import (
    _THROUGHPUT_UNROLL_FACTORS,
    KernelKind,
    KernelRequest,
    KernelSpec,
)


@dataclass(frozen=True, slots=True)
class ThroughputDomainResult:
    measurements: tuple[ThroughputMeasurement, ...]
    best: ThroughputResult | None


@dataclass(frozen=True, slots=True)
class ThroughputResults:
    image_build_info: DeviceImageBuildInfo
    exclusions: tuple[ThroughputExcludedLaunch, ...]
    domains: dict[str, ThroughputDomainResult]


def validate_results(
    results: tuple[ThroughputResults, ...],
    *,
    device_info: DeviceInfo,
    domain_names: Collection[str],
    generator_descriptions: Mapping[str, str],
) -> None:
    for library in results:
        if library.domains.keys() != set(domain_names):
            raise ValueError(
                "throughput libraries must have results for all selected domains"
            )
        for domain, result in library.domains.items():
            if domain not in generator_descriptions:
                raise ValueError(f"domain {domain!r} has no generator description")
            for measurement in result.measurements:
                if measurement.device_info != device_info:
                    raise ValueError("measurement device does not match device_info")
                if measurement.domain != domain:
                    raise ValueError(
                        "throughput measurement domain does not match its result"
                    )


def run(
    context: DeviceContext,
    function: FunctionConfig,
    image_build_infos: Sequence[DeviceImageBuildInfo],
    input_generators: Mapping[str, InputGenerator],
    metric: ThroughputConfig,
) -> tuple[ThroughputResults, ...]:
    kind = KernelKind.THROUGHPUT
    spec = KernelSpec.from_config(function, kind=kind)
    device_info = context.device_info
    results = []
    for image_build_info in image_build_infos:
        image = image_build_info.image
        module = context.load_module(image)
        exclusions = []
        benchmarks = []
        for unroll_factor in _THROUGHPUT_UNROLL_FACTORS:
            request = KernelRequest(
                image.library, function.name, kind, unroll_factor=unroll_factor
            )
            kernel = module.get_kernel(request.symbol, spec=spec)
            launch = try_plan_throughput_launch(
                device_info, kernel, request, batch_size=metric.batch_size
            )
            if isinstance(launch, ThroughputExcludedLaunch):
                exclusions.append(launch)
                continue
            benchmarks.append(
                ThroughputBenchmark(
                    context,
                    kernel,
                    launch,
                    sample_size=metric.sample.size,
                    timeout_s=metric.sample.timeout_s,
                    warmup_runs=metric.warmup_runs,
                    min_clock_ratio=metric.throttle.min_clock_ratio,
                    initial_delay_s=metric.throttle.initial_delay_s,
                )
            )

        domains = {}
        for domain, generator in input_generators.items():
            measurements = tuple(
                benchmark.measure(
                    domain, generator.batches(batch_size=metric.batch_size)
                )
                for benchmark in benchmarks
            )
            best = max(
                (
                    result
                    for measurement in measurements
                    if (result := analyze(measurement)) is not None
                ),
                key=lambda result: result.mean,
                default=None,
            )
            domains[domain] = ThroughputDomainResult(measurements, best)
        results.append(ThroughputResults(image_build_info, tuple(exclusions), domains))

    return tuple(results)
