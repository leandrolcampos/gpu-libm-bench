# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Math library selection for benchmark experiments."""

from pathlib import Path

from gpu_libm_bench.compiler import _paths
from gpu_libm_bench.compiler.images import configure_kernels, read_registered_functions
from gpu_libm_bench.compiler.libraries import (
    LibraryUnavailableError,
    ResolvedLibrary,
    resolve_library,
)
from gpu_libm_bench.compiler.targets import CompilationTarget
from gpu_libm_bench.compiler.toolchains import CUDAToolkit, LLVMToolchain
from gpu_libm_bench.configs import ExperimentConfig
from gpu_libm_bench.kernels import KernelKind, KernelSpec


def select_libraries(
    config: ExperimentConfig,
    *,
    llvm: LLVMToolchain,
    target: CompilationTarget,
    sdk: CUDAToolkit,
    build_root: Path | None = None,
) -> tuple[ResolvedLibrary, ...]:
    if target.backend != config.backend.name:
        raise ValueError("target backend does not match the experiment configuration")

    if config.metric.name == "latency":
        KernelSpec.from_config(config.function, kind=KernelKind.LATENCY)
    else:
        raise ValueError(f"unsupported experiment metric: {config.metric.name!r}")

    explicit = config.run.libraries is not None
    names = config.libraries if config.run.libraries is None else config.run.libraries
    if build_root is None:
        build_root = _paths.build_root() / "kernels" / "libraries"

    selected = []
    for name in names:
        library_config = config.libraries[name]
        try:
            bitcode_config = library_config.backends.get(target.backend)
            if bitcode_config is None:
                raise LibraryUnavailableError(
                    f"library {name!r} does not declare support "
                    f"for backend {target.backend!r}"
                )

            resolved_library = resolve_library(
                library=library_config.name,
                llvm=llvm,
                target=target,
                bitcode_path=Path(bitcode_config.bitcode_path)
                if bitcode_config.bitcode_path is not None
                else None,
                required_symbols=(config.function.name,),
            )

            build = configure_kernels(
                resolved_library=resolved_library,
                target=target,
                llvm=llvm,
                sdk=sdk,
                build_directory=build_root
                / target.backend
                / target.architecture
                / name,
            )
            functions = read_registered_functions(build, library=name, target=target)
            if config.function.name not in functions:
                raise LibraryUnavailableError(
                    f"library {name!r} has no wrappers "
                    f"for function {config.function.name!r} "
                    f"on backend {target.backend!r}"
                )
        except LibraryUnavailableError as error:
            if explicit:
                raise LibraryUnavailableError(
                    f"library {name!r} is unavailable for {config.function.name!r}/"
                    f"{config.metric.name!r} on backend {target.backend!r}: {error}"
                ) from error
            continue

        selected.append(resolved_library)

    if not selected:
        raise LibraryUnavailableError(
            f"no math libraries are available for {config.function.name!r}/"
            f"{config.metric.name!r} on backend {target.backend!r}"
        )

    return tuple(selected)
