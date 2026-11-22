# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Select and resolve math libraries for the requested experiment."""

from gpu_libm_bench.compiler.libraries import (
    LibraryUnavailableError,
    ResolvedLibrary,
    resolve_library,
)
from gpu_libm_bench.compiler.targets import CompilationTarget
from gpu_libm_bench.compiler.toolchains import BackendSDK, LLVMToolchain
from gpu_libm_bench.configs import ExperimentConfig
from gpu_libm_bench.kernels import KernelKind, KernelSpec


def select_libraries(
    config: ExperimentConfig,
    *,
    llvm: LLVMToolchain,
    target: CompilationTarget,
    backend_sdk: BackendSDK,
) -> tuple[ResolvedLibrary, ...]:
    if target.backend != config.backend.name:
        raise ValueError("target backend does not match the experiment configuration")

    if config.metric.name == "latency":
        KernelSpec.from_config(config.function, kind=KernelKind.LATENCY)
    elif config.metric.name == "throughput":
        KernelSpec.from_config(config.function, kind=KernelKind.THROUGHPUT)
    else:
        raise ValueError(f"unsupported experiment metric: {config.metric.name!r}")

    explicit = config.run.libraries is not None
    libraries = (
        config.libraries if config.run.libraries is None else config.run.libraries
    )
    selected = []
    for library in libraries:
        library_config = config.libraries[library]
        try:
            if config.function.name not in library_config.functions:
                raise LibraryUnavailableError(
                    f"library {library!r} does not declare function "
                    f"{config.function.name!r}"
                )

            resolved_library = resolve_library(
                library_config=library_config,
                llvm=llvm,
                target=target,
                backend_sdk=backend_sdk,
            )
        except LibraryUnavailableError as error:
            if explicit:
                raise LibraryUnavailableError(
                    f"library {library!r} is unavailable for {config.function.name!r}/"
                    f"{config.metric.name!r} on backend "
                    f"{target.backend.value!r}: {error}"
                ) from error
            continue

        selected.append(resolved_library)

    if not selected:
        raise LibraryUnavailableError(
            f"no math libraries are available for {config.function.name!r}/"
            f"{config.metric.name!r} on backend {target.backend.value!r}"
        )

    return tuple(selected)
