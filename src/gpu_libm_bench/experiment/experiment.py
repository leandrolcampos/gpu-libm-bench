# ===----------------------------------------------------------------------=== #
# Copyright 2026 Leandro Lacerda Campos
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Benchmark experiment orchestration and result validation."""

import platform
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from gpu_libm_bench import inputs
from gpu_libm_bench.compiler import paths
from gpu_libm_bench.compiler.backends import (
    BackendModuleNotFoundError,
    build_backend,
    get_backend_module,
)
from gpu_libm_bench.compiler.images import (
    DeviceImageBuildInfo,
    build_kernels,
    read_device_image_build_info,
)
from gpu_libm_bench.compiler.libraries import ResolvedLibrary
from gpu_libm_bench.compiler.targets import CompilationTarget, resolve_target
from gpu_libm_bench.compiler.toolchains import (
    BackendSDK,
    LLVMToolchain,
    resolve_backend_sdk,
    resolve_llvm,
)
from gpu_libm_bench.configs import ExperimentConfig, LatencyConfig, ThroughputConfig
from gpu_libm_bench.devices import DeviceContext, DeviceInfo

from . import _latency, _libraries, _throughput


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    config: ExperimentConfig
    device_info: DeviceInfo
    environment: dict[str, str]
    generator_descriptions: dict[str, str]
    results: tuple[_latency.LatencyResults | _throughput.ThroughputResults, ...]

    def __post_init__(self) -> None:
        if self.device_info.backend != self.config.backend.name:
            raise ValueError("result device backend does not match its configuration")

        libraries = [result.image_build_info.image.library for result in self.results]
        if len(set(libraries)) != len(libraries):
            raise ValueError(
                "experiment results must contain at most one result per library"
            )

        for result in self.results:
            image = result.image_build_info.image
            if image.library not in self.config.libraries:
                raise ValueError(f"result has an unknown library: {image.library!r}")

            if (
                image.target.backend != self.device_info.backend
                or image.target.architecture != self.device_info.architecture
                or image.target.target_triple != self.config.backend.target_triple
            ):
                raise ValueError(
                    "result device image target does not match the experiment target"
                )

        domain_names = (
            self.config.function.domains
            if self.config.run.domains is None
            else self.config.run.domains
        )
        match self.config.metric.name:
            case "latency":
                _latency.validate_results(
                    self.results_for(_latency.LatencyResults),
                    device_info=self.device_info,
                    domain_names=domain_names,
                    generator_descriptions=self.generator_descriptions,
                )
            case "throughput":
                _throughput.validate_results(
                    self.results_for(_throughput.ThroughputResults),
                    device_info=self.device_info,
                    domain_names=domain_names,
                    generator_descriptions=self.generator_descriptions,
                )
            case name:
                raise ValueError(f"unsupported experiment metric: {name!r}")

    def results_for[T: (_latency.LatencyResults, _throughput.ThroughputResults)](
        self, result_type: type[T]
    ) -> tuple[T, ...]:
        results = tuple(
            result for result in self.results if isinstance(result, result_type)
        )
        if len(results) != len(self.results):
            raise ValueError("experiment result types must match the metric")
        return results


class Experiment:
    def __init__(self, config: ExperimentConfig) -> None:
        self._config = config
        metric = config.metric
        input_count: int | None
        match metric:
            case LatencyConfig(name="latency"):
                input_count = metric.sample.size
            case ThroughputConfig(name="throughput"):
                if metric.confirmation.repetitions < 2:
                    raise ValueError(
                        "metric.confirmation.repetitions must be at least 2"
                    )
                input_count = None
            case _:
                raise ValueError(f"unsupported experiment metric: {metric.name!r}")

        rng = np.random.Generator(np.random.PCG64DXSM(config.run.seed))
        inputs_rng, self.ordering_rng = rng.spawn(2)

        function = config.function
        names = function.domains if config.run.domains is None else config.run.domains
        self._input_generators: dict[str, inputs.InputGenerator] = {}
        self._confirmation_generators: dict[str, tuple[inputs.InputGenerator, ...]] = {}
        stream_names = sorted(function.domains) if input_count is None else names
        domain_rngs = dict(
            zip(stream_names, inputs_rng.spawn(len(stream_names)), strict=True)
        )
        for name in names:
            domain_rng = domain_rngs[name]
            generator_config = function.domains[name].generator
            if isinstance(metric, ThroughputConfig):
                domain_rng, *confirmation_rngs = domain_rng.spawn(
                    1 + metric.confirmation.repetitions
                )
                self._confirmation_generators[name] = tuple(
                    inputs.InputGenerator.from_config(
                        generator_config,
                        function.signature,
                        sample_size=None,
                        domain=name,
                        rng=confirmation_rng,
                    )
                    for confirmation_rng in confirmation_rngs
                )
            self._input_generators[name] = inputs.InputGenerator.from_config(
                generator_config,
                function.signature,
                sample_size=input_count,
                domain=name,
                rng=domain_rng,
            )
        self.generator_descriptions = {
            name: str(generator) for name, generator in self._input_generators.items()
        }

    def run(self) -> ExperimentResult:
        config = self._config
        metric = config.metric
        backend = config.backend
        backend_sdk = resolve_backend_sdk(
            backend=backend.name,
            root=Path(backend.sdk_root) if backend.sdk_root is not None else None,
        )
        self._prepare_backend(backend_sdk)

        with DeviceContext(backend.device_id, backend=backend.name) as context:
            device_info = context.device_info
            llvm = resolve_llvm(
                Path(config.llvm.root) if config.llvm.root is not None else None
            )
            target = resolve_target(
                backend=backend.name,
                target_triple=backend.target_triple,
                architecture=backend.architecture,
                llvm=llvm,
                device_info=device_info,
                backend_sdk=backend_sdk,
            )

            image_build_infos = self._prepare_images(
                llvm=llvm, target=target, backend_sdk=backend_sdk
            )

            # TODO: Replace this temporary message with progress reporting.
            print("Running benchmark. This may take a few minutes...", file=sys.stderr)
            results: tuple[_latency.LatencyResults | _throughput.ThroughputResults, ...]
            match metric:
                case LatencyConfig():
                    results = _latency.run(
                        context,
                        config.function,
                        image_build_infos,
                        tuple(self._input_generators.values()),
                        sample_size=metric.sample.size,
                        timeout_s=metric.sample.timeout_s,
                        chain_length=metric.chain_length,
                        warmup_runs=metric.warmup_runs,
                        throttle=metric.throttle,
                        should_discount_chain_cost=metric.should_discount_chain_cost,
                    )
                case ThroughputConfig():
                    results = _throughput.run(
                        context,
                        config.function,
                        image_build_infos,
                        self._input_generators,
                        self._confirmation_generators,
                        metric,
                    )
                case _:
                    raise ValueError(f"unsupported experiment metric: {metric.name!r}")

        effective_config = replace(
            config,
            backend=replace(
                backend,
                architecture=target.architecture,
                sdk_root=str(backend_sdk.root),
            ),
            llvm=replace(config.llvm, root=str(llvm.root)),
        )

        return ExperimentResult(
            config=effective_config,
            device_info=device_info,
            environment={
                "Python": platform.python_version(),
                "NumPy": np.__version__,
                "Platform": platform.platform(),
            },
            generator_descriptions=self.generator_descriptions,
            results=results,
        )

    def _prepare_backend(self, backend_sdk: BackendSDK) -> None:
        backend = self._config.backend

        try:
            get_backend_module(backend.name)
        except BackendModuleNotFoundError:
            build_backend(backend.name, backend_sdk=backend_sdk)

    def _prepare_images(
        self,
        *,
        llvm: LLVMToolchain,
        target: CompilationTarget,
        backend_sdk: BackendSDK,
    ) -> tuple[DeviceImageBuildInfo, ...]:
        resolved_libraries = _libraries.select_libraries(
            self._config, llvm=llvm, target=target, backend_sdk=backend_sdk
        )

        image_build_infos = []
        for resolved_library in (ResolvedLibrary("support"), *resolved_libraries):
            image = build_kernels(
                resolved_library=resolved_library,
                target=target,
                llvm=llvm,
                backend_sdk=backend_sdk,
                build_directory=paths.image_build_directory(
                    resolved_library.name,
                    backend=target.backend,
                    architecture=target.architecture,
                ),
            )
            if resolved_library.name != "support":
                image_build_infos.append(read_device_image_build_info(image))

        return tuple(image_build_infos)
