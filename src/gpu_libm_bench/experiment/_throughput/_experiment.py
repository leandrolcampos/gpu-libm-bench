# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Throughput execution and result aggregation for experiments."""

from collections.abc import Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass

from gpu_libm_bench.benchmarks.throughput import (
    ThroughputBenchmark,
    ThroughputMeasurement,
    ThroughputPlannedLaunch,
    ThroughputResult,
    ThroughputSearchPlan,
    analyze,
    plan_throughput_search,
)
from gpu_libm_bench.compiler.images import DeviceImageBuildInfo
from gpu_libm_bench.configs import FunctionConfig, ThroughputConfig
from gpu_libm_bench.devices import DeviceContext, DeviceInfo, DeviceKernel
from gpu_libm_bench.inputs import InputGenerator
from gpu_libm_bench.kernels import (
    _THROUGHPUT_UNROLL_FACTORS,
    KernelKind,
    KernelRequest,
    KernelSpec,
)

from ._confirmation import ThroughputConfirmation, confirm_winner


@dataclass(frozen=True, slots=True)
class ThroughputDomainResult:
    search_measurements: tuple[ThroughputMeasurement, ...]
    search_winner: ThroughputResult | None
    confirmation: ThroughputConfirmation | None

    @property
    def reported_result(self) -> ThroughputResult | None:
        confirmation = self.confirmation
        if confirmation is not None and confirmation.unconfirmed_reason is None:
            return confirmation.winner_estimate
        return self.search_winner

    def iter_measurements(self) -> Iterator[ThroughputMeasurement]:
        yield from self.search_measurements
        if self.confirmation is not None:
            yield from self.confirmation.winner_measurements
            for contrast in self.confirmation.contrasts:
                yield from contrast.measurements


@dataclass(frozen=True, slots=True)
class ThroughputResults:
    image_build_info: DeviceImageBuildInfo
    plan: ThroughputSearchPlan
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
            for measurement in result.iter_measurements():
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
    confirmation_generators: Mapping[str, tuple[InputGenerator, ...]],
    metric: ThroughputConfig,
) -> tuple[ThroughputResults, ...]:
    kind = KernelKind.THROUGHPUT
    spec = KernelSpec.from_config(function, kind=kind)
    results = []
    for image_build_info in image_build_infos:
        image = image_build_info.image
        module = context.load_module(image)
        requests = (
            KernelRequest(image.library, function.name, kind, unroll_factor=factor)
            for factor in _THROUGHPUT_UNROLL_FACTORS
        )
        kernels = {
            request: module.get_kernel(request.symbol, spec=spec)
            for request in requests
        }
        plan = plan_throughput_search(
            context.device_info, kernels, max_candidates=metric.search.max_candidates
        )
        domains = {
            domain: _measure_domain(
                context,
                kernels,
                plan,
                metric,
                domain,
                generator,
                confirmation_generators[domain],
            )
            for domain, generator in input_generators.items()
        }
        results.append(ThroughputResults(image_build_info, plan, domains))

    return tuple(results)


def _measure_domain(
    context: DeviceContext,
    kernels: Mapping[KernelRequest, DeviceKernel],
    plan: ThroughputSearchPlan,
    metric: ThroughputConfig,
    domain: str,
    search_generator: InputGenerator,
    confirmation_generators: tuple[InputGenerator, ...],
) -> ThroughputDomainResult:
    def measure_launch(
        launch: ThroughputPlannedLaunch, generator: InputGenerator
    ) -> ThroughputMeasurement:
        benchmark = ThroughputBenchmark(
            context,
            kernels[launch.request],
            launch,
            sample_size=metric.sample.size,
            timeout_s=metric.sample.timeout_s,
            warmup_runs=metric.warmup_runs,
            min_clock_ratio=metric.throttle.min_clock_ratio,
            initial_delay_s=metric.throttle.initial_delay_s,
        )
        return benchmark.measure(
            domain,
            generator.batches(
                batch_size=launch.batch_size, batch_count=metric.sample.size
            ),
        )

    search_measurements = tuple(
        measure_launch(launch, search_generator) for launch in plan.candidates
    )
    search_winner = max(
        (
            result
            for measurement in search_measurements
            if (result := analyze(measurement)) is not None
        ),
        key=lambda result: result.mean,
        default=None,
    )
    confirmation = (
        confirm_winner(
            search_winner,
            plan.eligible,
            confirmation_generators,
            measure_launch,
            search_complete=all(
                measurement.complete for measurement in search_measurements
            ),
        )
        if search_winner is not None
        else None
    )
    return ThroughputDomainResult(search_measurements, search_winner, confirmation)
