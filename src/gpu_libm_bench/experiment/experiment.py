# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Benchmark experiment orchestration."""

import platform
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from gpu_libm_bench import inputs
from gpu_libm_bench.compiler import paths
from gpu_libm_bench.compiler.backends import (
    BackendNotFoundError,
    build_backend,
    get_backend,
)
from gpu_libm_bench.compiler.images import (
    DeviceImageBuildInfo,
    build_kernels,
    read_device_image_build_info,
)
from gpu_libm_bench.compiler.libraries import ResolvedLibrary
from gpu_libm_bench.compiler.targets import CompilationTarget, resolve_target
from gpu_libm_bench.compiler.toolchains import (
    CUDAToolkit,
    LLVMToolchain,
    resolve_llvm,
    resolve_sdk,
)
from gpu_libm_bench.configs import ExperimentConfig, LatencyConfig
from gpu_libm_bench.devices import DeviceContext, DeviceInfo

from . import _latency, _libraries


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    config: ExperimentConfig
    device_info: DeviceInfo
    environment: dict[str, str]
    generator_descriptions: dict[str, str]
    results: tuple[_latency.LatencyResults, ...]

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

        if self.config.metric.name == "latency":
            _latency.validate_results(
                self.results,
                device_info=self.device_info,
                domain_names=self.config.function.domains
                if self.config.run.domains is None
                else self.config.run.domains,
                generator_descriptions=self.generator_descriptions,
            )
        else:
            raise ValueError(
                f"unsupported experiment metric: {self.config.metric.name!r}"
            )


class Experiment:
    def __init__(self, config: ExperimentConfig) -> None:
        self._config = config
        metric = config.metric
        if not isinstance(metric, LatencyConfig) or metric.name != "latency":
            raise ValueError(f"unsupported experiment metric: {metric.name!r}")

        rng = np.random.Generator(np.random.PCG64DXSM(config.run.seed))
        inputs_rng, self.ordering_rng = rng.spawn(2)

        function = config.function
        names = function.domains if config.run.domains is None else config.run.domains
        self._input_generators = {
            name: inputs.InputGenerator.from_config(
                function.domains[name].generator,
                function.signature,
                sample_size=metric.sample_size,
                domain=name,
                rng=domain_rng,
            )
            for name, domain_rng in zip(
                names, inputs_rng.spawn(len(names)), strict=True
            )
        }
        self.generator_descriptions = {
            name: str(generator) for name, generator in self._input_generators.items()
        }

    def run(self) -> ExperimentResult:
        config = self._config
        metric = config.metric
        if not isinstance(metric, LatencyConfig) or metric.name != "latency":
            raise ValueError(f"unsupported experiment metric: {metric.name!r}")

        backend = config.backend
        sdk = resolve_sdk(
            backend=backend.name,
            root=Path(backend.sdk_root) if backend.sdk_root is not None else None,
        )
        self._prepare_backend(sdk)

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
                sdk=sdk,
            )

            image_build_infos = self._prepare_images(llvm=llvm, target=target, sdk=sdk)

            # TODO: Replace this temporary message with progress reporting.
            print("Running benchmark. This may take a few minutes...", file=sys.stderr)
            results = _latency.run(
                context,
                config.function,
                image_build_infos,
                tuple(self._input_generators.values()),
                sample_size=metric.sample_size,
                chain_length=metric.chain_length,
                repetitions=metric.repetitions,
                warmup_runs=metric.warmup_runs,
            )

        effective_config = replace(
            config,
            backend=replace(
                backend, architecture=target.architecture, sdk_root=str(sdk.root)
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

    def _prepare_backend(self, sdk: CUDAToolkit) -> None:
        backend = self._config.backend
        try:
            get_backend(backend.name)
        except BackendNotFoundError:
            build_backend(backend.name, sdk=sdk)

    def _prepare_images(
        self,
        *,
        llvm: LLVMToolchain,
        target: CompilationTarget,
        sdk: CUDAToolkit,
    ) -> tuple[DeviceImageBuildInfo, ...]:
        resolved_libraries = _libraries.select_libraries(
            self._config, llvm=llvm, target=target, sdk=sdk
        )

        image_build_infos = []
        for resolved_library in (ResolvedLibrary("support"), *resolved_libraries):
            image = build_kernels(
                resolved_library=resolved_library,
                target=target,
                llvm=llvm,
                sdk=sdk,
                build_directory=paths.image_build_directory(
                    resolved_library.library,
                    backend=target.backend,
                    architecture=target.architecture,
                ),
            )
            if resolved_library.library != "support":
                image_build_infos.append(read_device_image_build_info(image))

        return tuple(image_build_infos)
