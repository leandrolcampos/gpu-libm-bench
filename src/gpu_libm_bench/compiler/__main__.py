# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Command-line configuration and compilation of native backends and GPU kernels."""

import argparse
from pathlib import Path
from typing import cast

from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from gpu_libm_bench.configs import (
    BackendConfig,
    LibraryConfig,
    LLVMConfig,
    register_config_schema,
)
from gpu_libm_bench.devices import DeviceContext

from .backends import build_backend, configure_backend
from .images import build_kernels, configure_kernels
from .libraries import resolve_library
from .targets import resolve_target
from .toolchains import resolve_llvm, resolve_sdk


def main() -> None:
    parser = argparse.ArgumentParser(description="Configure or build native GPU code.")
    parser.add_argument("project", choices=("backend", "kernels"))
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--configure-only", action="store_true")
    parser.add_argument("--build-tests", action="store_true")
    parser.add_argument(
        "overrides", nargs="*", help="Hydra compilation configuration overrides"
    )
    arguments = parser.parse_intermixed_args()

    register_config_schema()
    excluded = ["~function", "~metric"]

    with initialize_config_module(config_module="gpu_libm_bench.configs.yaml"):
        config = compose(
            config_name="config", overrides=[*excluded, *arguments.overrides]
        )

    backend = cast("BackendConfig", OmegaConf.to_object(config.backend))
    sdk = resolve_sdk(
        backend=backend.name,
        root=Path(backend.sdk_root) if backend.sdk_root is not None else None,
    )

    if arguments.project == "backend":
        compile_backend = (
            configure_backend if arguments.configure_only else build_backend
        )
        compile_backend(
            backend.name,
            sdk=sdk,
            build_directory=arguments.build_dir,
            build_tests=arguments.build_tests,
        )
        return

    llvm_config = cast("LLVMConfig", OmegaConf.to_object(config.llvm))
    selected = config.run.libraries
    if selected is None or len(selected) != 1:
        raise ValueError(
            "kernel compilation requires exactly one name in run.libraries"
        )
    name = selected[0]
    if name not in config.libraries:
        raise ValueError(f"run.libraries contains an unknown library: {name!r}")
    library_config = cast("LibraryConfig", OmegaConf.to_object(config.libraries[name]))
    if backend.name not in library_config.backends:
        raise ValueError(
            f"library {library_config.name!r} does not support backend {backend.name!r}"
        )

    llvm = resolve_llvm(
        Path(llvm_config.root) if llvm_config.root is not None else None
    )
    device_info = (
        DeviceContext.get_device_info(backend.device_id, backend=backend.name)
        if backend.architecture is None
        else None
    )
    target = resolve_target(
        backend=backend.name,
        target_triple=backend.target_triple,
        architecture=backend.architecture,
        llvm=llvm,
        device_info=device_info,
        sdk=sdk,
    )
    bitcode = library_config.backends[backend.name].bitcode_path
    library = resolve_library(
        library=library_config.name,
        llvm=llvm,
        target=target,
        bitcode_path=Path(bitcode) if bitcode is not None else None,
    )
    compile_kernels = configure_kernels if arguments.configure_only else build_kernels
    compile_kernels(
        library=library,
        target=target,
        llvm=llvm,
        sdk=sdk,
        build_directory=arguments.build_dir,
        build_tests=arguments.build_tests,
    )


if __name__ == "__main__":
    main()
