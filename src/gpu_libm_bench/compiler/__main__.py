# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Command-line configuration and compilation of native backends and GPU kernels."""

import argparse
import sys
from collections.abc import Sequence
from itertools import groupby
from pathlib import Path
from typing import cast

from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra.core.override_parser.types import Override

from gpu_libm_bench.configs import (
    BackendConfig,
    BuildConfig,
    LibraryConfig,
    register_config_schema,
    validate_build_config,
)
from gpu_libm_bench.devices import DeviceContext, DeviceInfo

from ._cmake import CMakeBuild
from ._commands import run_tool
from .backends import BackendNotFoundError, build_backend, configure_backend
from .images import build_kernels, configure_kernels
from .libraries import LibraryUnavailableError, resolve_library
from .targets import CompilationTarget, TargetUnavailableError, resolve_target
from .toolchains import (
    CUDAToolkit,
    LLVMToolchain,
    SDKUnavailableError,
    resolve_llvm,
    resolve_sdk,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Configure or build all available native GPU backends or kernels.",
        epilog=(
            "Filter with backend=<name> and/or library=<name>. "
            "Kernel builds include all registered functions."
        ),
    )
    parser.add_argument("project", choices=("backend", "kernels"))
    parser.add_argument(
        "--build-root", type=Path, help="Root directory for native builds."
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--configure-only", action="store_true")
    mode.add_argument(
        "--tidy", action="store_true", help="Configure kernels and run clang-tidy."
    )

    parser.add_argument("--build-tests", action="store_true")
    parser.add_argument(
        "overrides", nargs="*", help="Hydra compilation configuration overrides"
    )

    arguments = parser.parse_intermixed_args()
    if arguments.tidy and arguments.project != "kernels":
        parser.error("--tidy is supported only for kernels")

    register_config_schema()
    parsed = OverridesParser.create().parse_overrides(arguments.overrides)

    with initialize_config_module(config_module="gpu_libm_bench.configs.yaml"):
        loader = GlobalHydra.instance().config_loader()
        backends, explicit_backend = _selected_options(
            "backend", parsed, loader.get_group_options("backend")
        )

        if arguments.project == "kernels":
            libraries, explicit_library = _selected_options(
                "library", parsed, loader.get_group_options("library")
            )
        else:
            if any(override.key_or_group == "library" for override in parsed):
                parser.error("library selection is supported only for kernels")
            libraries = []
            explicit_library = False

        overrides = [
            cast("str", override.input_line)
            for override in parsed
            if override.key_or_group not in {"backend", "library"}
        ]

        configs: list[BuildConfig] = []
        for backend in backends:
            for library in libraries if arguments.project == "kernels" else [None]:
                config = compose(
                    config_name="build_config",
                    overrides=[
                        f"backend={backend}",
                        *([f"library={library}"] if library is not None else []),
                        *overrides,
                    ],
                )
                configs.append(validate_build_config(config))

    if arguments.project == "backend":
        _compile_backends(configs, arguments, explicit_backend=explicit_backend)
    else:
        _compile_kernels(
            configs,
            arguments,
            explicit_backend=explicit_backend,
            explicit_library=explicit_library,
        )


def _selected_options(
    group: str, overrides: Sequence[Override], available: list[str]
) -> tuple[list[str], bool]:
    selection = next(
        (
            override
            for override in reversed(overrides)
            if override.key_or_group == group
        ),
        None,
    )

    if selection is None:
        return available, False
    if selection.is_delete() or selection.is_sweep_override():
        raise ValueError(f"{group} must be a single name")

    name = selection.value()
    if name is None:
        return available, False
    if not isinstance(name, str):
        raise ValueError(f"{group} must be a single name")
    if name not in available:
        raise ValueError(f"unknown {group}: {name!r}")
    return [name], True


def _compile_backends(
    configs: Sequence[BuildConfig],
    arguments: argparse.Namespace,
    *,
    explicit_backend: bool,
) -> None:
    build_root = arguments.build_root
    compile_backend = configure_backend if arguments.configure_only else build_backend
    completed = 0

    for config in configs:
        backend = config.backend
        try:
            sdk = resolve_sdk(
                backend=backend.name,
                root=Path(backend.sdk_root) if backend.sdk_root is not None else None,
            )
        except SDKUnavailableError as error:
            if explicit_backend:
                raise
            print(f"Skipping {backend.name}: {error}", file=sys.stderr)
            continue

        compile_backend(
            backend.name,
            sdk=sdk,
            build_directory=build_root / backend.name
            if build_root is not None
            else None,
            build_tests=arguments.build_tests,
        )
        completed += 1

    if not completed:
        raise RuntimeError("no native backends are available for compilation")


def _compile_kernels(
    configs: Sequence[BuildConfig],
    arguments: argparse.Namespace,
    *,
    explicit_backend: bool,
    explicit_library: bool,
) -> None:
    llvm: LLVMToolchain | None = None
    completed = False
    requested: str | None = None

    for _, group in groupby(configs, key=lambda config: config.backend.name):
        choices = tuple(group)
        backend = choices[0].backend
        libraries = _select_libraries(choices)
        selected_library = choices[0].library
        if explicit_library and selected_library is not None:
            requested = selected_library.name

        if not libraries:
            if explicit_backend and explicit_library:
                raise LibraryUnavailableError(
                    f"library {requested!r} does not support backend {backend.name!r}"
                )
            continue

        try:
            sdk = resolve_sdk(
                backend=backend.name,
                root=Path(backend.sdk_root) if backend.sdk_root is not None else None,
            )
            if llvm is None:
                llvm_config = choices[0].llvm
                llvm = resolve_llvm(
                    Path(llvm_config.root) if llvm_config.root is not None else None
                )

            device_info = _get_device_info(backend, sdk=sdk)
            target = resolve_target(
                backend=backend.name,
                target_triple=backend.target_triple,
                architecture=backend.architecture,
                llvm=llvm,
                device_info=device_info,
                sdk=sdk,
            )
        except (SDKUnavailableError, TargetUnavailableError) as error:
            if explicit_backend:
                raise
            print(f"Skipping {backend.name}: {error}", file=sys.stderr)
            continue

        for library in libraries:
            try:
                _compile_library_kernels(
                    library,
                    arguments,
                    llvm=llvm,
                    target=target,
                    sdk=sdk,
                )
            except LibraryUnavailableError as error:
                if explicit_backend and explicit_library:
                    raise
                print(
                    f"Skipping {library.name}/{backend.name}: {error}", file=sys.stderr
                )
                continue

            completed = True

    if not completed:
        if requested is not None:
            raise LibraryUnavailableError(
                f"selected math library is unavailable: {requested}"
            )
        raise LibraryUnavailableError(
            "no library/backend combinations are available for kernel compilation"
        )


def _select_libraries(configs: Sequence[BuildConfig]) -> list[LibraryConfig]:
    libraries = []
    names: set[str] = set()

    for config in configs:
        library = config.library
        if library is None:
            raise ValueError("kernel compilation requires a library")
        if library.name in names:
            raise ValueError("selected libraries must have distinct names")
        names.add(library.name)

        if config.backend.name in library.backends:
            libraries.append(library)

    return libraries


def _get_device_info(backend: BackendConfig, *, sdk: CUDAToolkit) -> DeviceInfo | None:
    if backend.architecture is not None:
        return None

    try:
        count = DeviceContext.device_count(backend=backend.name)
    except BackendNotFoundError:
        build_backend(backend.name, sdk=sdk)
        count = DeviceContext.device_count(backend=backend.name)

    if count == 0:
        raise TargetUnavailableError(
            f"cannot infer an architecture for {backend.name}: "
            "no devices found; set backend.architecture"
        )
    return DeviceContext.get_device_info(backend.device_id, backend=backend.name)


def _compile_library_kernels(
    library: LibraryConfig,
    arguments: argparse.Namespace,
    *,
    llvm: LLVMToolchain,
    target: CompilationTarget,
    sdk: CUDAToolkit,
) -> None:
    bitcode_path = library.backends[target.backend].bitcode_path
    resolved_library = resolve_library(
        library=library.name,
        llvm=llvm,
        target=target,
        bitcode_path=Path(bitcode_path) if bitcode_path is not None else None,
    )

    compile_kernels = (
        configure_kernels
        if arguments.configure_only or arguments.tidy
        else build_kernels
    )
    build_root = arguments.build_root
    build = compile_kernels(
        resolved_library=resolved_library,
        target=target,
        llvm=llvm,
        sdk=sdk,
        build_directory=build_root / target.backend / target.architecture / library.name
        if build_root is not None
        else None,
        build_tests=arguments.build_tests,
    )

    if arguments.tidy:
        configured_build = cast(CMakeBuild, build)
        run_tool(
            ["run-clang-tidy", "-quiet", "-p", configured_build.build_directory],
            stream=True,
            timeout=None,
        )


if __name__ == "__main__":
    main()
