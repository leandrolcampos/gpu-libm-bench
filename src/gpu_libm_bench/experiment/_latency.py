# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Latency execution and result aggregation for experiments."""

from collections.abc import Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass

from gpu_libm_bench import inputs
from gpu_libm_bench.benchmarks.latency import LatencyBenchmark, LatencyResult, analyze
from gpu_libm_bench.compiler.images import DeviceImageBuildInfo
from gpu_libm_bench.configs import FunctionConfig
from gpu_libm_bench.devices import DeviceContext, DeviceInfo
from gpu_libm_bench.kernels import KernelKind, KernelRequest, KernelSpec


@dataclass(frozen=True, slots=True)
class LatencyResults(Mapping[str, LatencyResult]):
    image_build_info: DeviceImageBuildInfo
    results: tuple[LatencyResult, ...]

    def __post_init__(self) -> None:
        domains = [result.measurement.inputs.domain for result in self.results]
        if len(set(domains)) != len(domains):
            raise ValueError(
                "latency results must contain at most one result per domain"
            )

    def __getitem__(self, domain: str) -> LatencyResult:
        for result in self.results:
            if result.measurement.inputs.domain == domain:
                return result
        raise KeyError(domain)

    def __iter__(self) -> Iterator[str]:
        return (result.measurement.inputs.domain for result in self.results)

    def __len__(self) -> int:
        return len(self.results)

    def validate_against(
        self,
        *,
        device_info: DeviceInfo,
        domain_names: Collection[str],
        generator_descriptions: Mapping[str, str],
    ) -> None:
        for result in self.results:
            if result.measurement.device_info != device_info:
                raise ValueError("measurement device does not match device_info")

            domain = result.measurement.inputs.domain
            if domain not in domain_names:
                raise ValueError(f"latency result has an unknown domain: {domain!r}")
            if domain not in generator_descriptions:
                raise ValueError(f"domain {domain!r} has no generator description")


def validate_results(
    results: tuple[LatencyResults, ...],
    *,
    device_info: DeviceInfo,
    domain_names: Collection[str],
    generator_descriptions: Mapping[str, str],
) -> None:
    domains = set(results[0]) if results else set()
    for library_result in results:
        library_result.validate_against(
            device_info=device_info,
            domain_names=domain_names,
            generator_descriptions=generator_descriptions,
        )
        if set(library_result) != domains:
            raise ValueError("latency libraries must have results for the same domains")


def run(
    context: DeviceContext,
    function: FunctionConfig,
    image_build_infos: Sequence[DeviceImageBuildInfo],
    input_generators: Sequence[inputs.InputGenerator],
    *,
    sample_size: int,
    chain_length: int,
    repetitions: int,
    warmup_runs: int,
) -> tuple[LatencyResults, ...]:
    spec = KernelSpec.from_config(function, kind=KernelKind.LATENCY)
    results = []
    for image_build_info in image_build_infos:
        image = image_build_info.image
        module = context.load_module(image)
        request = KernelRequest(image.library, function.name, KernelKind.LATENCY)
        kernel = module.get_kernel(request.symbol, spec=spec)
        benchmark = LatencyBenchmark(
            context,
            kernel,
            chain_length=chain_length,
            repetitions=repetitions,
            warmup_runs=warmup_runs,
        )

        results.append(
            LatencyResults(
                image_build_info,
                tuple(
                    analyze(benchmark.measure(batch))
                    for generator in input_generators
                    for batch in generator.batches(batch_size=sample_size)
                ),
            )
        )

    return tuple(results)
