# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Latency execution and result aggregation for experiments."""

from collections.abc import Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import partial

from gpu_libm_bench import inputs
from gpu_libm_bench.benchmarks.latency import (
    ChainCostEstimate,
    LatencyBenchmark,
    LatencyResult,
    analyze,
)
from gpu_libm_bench.compiler.images import DeviceImageBuildInfo
from gpu_libm_bench.configs import (
    ArgumentConfig,
    FunctionConfig,
    SignatureConfig,
    ThrottleConfig,
)
from gpu_libm_bench.devices import DeviceContext, DeviceInfo
from gpu_libm_bench.kernels import KernelKind, KernelRequest, KernelSpec


@dataclass(frozen=True, slots=True)
class LatencyResults(Mapping[str, LatencyResult]):
    image_build_info: DeviceImageBuildInfo
    results: tuple[LatencyResult, ...]
    chain_cost: ChainCostEstimate | None = None

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
    timeout_s: float,
    chain_length: int,
    warmup_runs: int,
    throttle: ThrottleConfig,
    should_discount_chain_cost: bool,
) -> tuple[LatencyResults, ...]:
    kind = KernelKind.LATENCY
    spec = KernelSpec.from_config(function, kind=kind)
    identity = _get_identity_config(function.signature.return_type)
    identity_spec = KernelSpec.from_config(identity, kind=kind)

    benchmark_for = partial(
        LatencyBenchmark,
        context,
        timeout_s=timeout_s,
        chain_length=chain_length,
        warmup_runs=warmup_runs,
        min_clock_ratio=throttle.min_clock_ratio,
        initial_delay_s=throttle.initial_delay_s,
    )

    chain_cost = None
    results = []
    for image_build_info in image_build_infos:
        image = image_build_info.image
        module = context.load_module(image)
        if chain_cost is None:
            control_request = KernelRequest(image.library, identity.name, kind)
            control = module.get_kernel(control_request.symbol, spec=identity_spec)
            chain_cost = benchmark_for(control).estimate_chain_cost()

        request = KernelRequest(image.library, function.name, kind)
        kernel = module.get_kernel(request.symbol, spec=spec)
        benchmark = benchmark_for(kernel)

        try:
            measurements = tuple(
                analyze(
                    benchmark.measure(
                        batch,
                        record_clocks=should_discount_chain_cost
                        and chain_cost.cycles_per_call is not None,
                    ),
                    chain_cost=chain_cost,
                    should_discount_chain_cost=should_discount_chain_cost,
                )
                for generator in input_generators
                for batch in generator.batches(batch_size=sample_size)
            )
        except TimeoutError as error:
            raise TimeoutError(f"{function.name} ({image.library}): {error}") from error

        results.append(LatencyResults(image_build_info, measurements, chain_cost))

    return tuple(results)


def _get_identity_config(return_type: str) -> FunctionConfig:
    return_type = return_type.strip()
    name = {"_Float16": "idf16", "float": "idf", "double": "id"}[return_type]
    signature = SignatureConfig(
        return_type, [ArgumentConfig("x", return_type, "input")]
    )
    return FunctionConfig(name=name, signature=signature, domains={})
