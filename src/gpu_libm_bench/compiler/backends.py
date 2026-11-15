# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Native backend compilation and loading."""

import sys
import sysconfig
from importlib import import_module
from importlib.machinery import EXTENSION_SUFFIXES
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

from gpu_libm_bench.backends import Backend

from . import paths
from ._cmake import CMakeBuild
from .toolchains import (
    BackendSDK,
    CUDAToolkit,
    ROCmCoreSDK,
    validate_backend_sdk_compatibility,
)


class BackendModuleNotFoundError(RuntimeError):
    pass


def build_backend(
    backend: Backend | str,
    *,
    backend_sdk: BackendSDK,
    build_directory: Path | None = None,
    build_tests: bool = False,
) -> Path:
    build = configure_backend(
        backend,
        backend_sdk=backend_sdk,
        build_directory=build_directory,
        build_tests=build_tests,
    )
    build.build()

    directory = build.build_directory.expanduser().resolve() / "python"
    for suffix in EXTENSION_SUFFIXES:
        path = directory / f"_{backend}{suffix}"
        if path.is_file():
            return path

    raise FileNotFoundError(
        f"backend build did not produce an extension in {directory}"
    )


def configure_backend(
    backend: Backend | str,
    *,
    backend_sdk: BackendSDK,
    build_directory: Path | None = None,
    build_tests: bool = False,
) -> CMakeBuild:
    validate_backend_sdk_compatibility(backend, backend_sdk)
    backend = Backend(backend)

    match backend_sdk:
        case CUDAToolkit():
            backend_sdk_arguments = (f"-DCUDAToolkit_ROOT={backend_sdk.root}",)
            backend_sdk_files = (backend_sdk.ptxas,)
        case ROCmCoreSDK():
            raise NotImplementedError(
                "hip backend with ROCm Core SDK is not implemented yet"
            )

    pybind11 = import_module("pybind11")
    project = paths.source_root()

    build = CMakeBuild(
        project / "backends" / backend,
        build_directory
        if build_directory is not None
        else paths.backend_build_directory(backend),
        arguments=(
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DBUILD_TESTING={'ON' if build_tests else 'OFF'}",
            *backend_sdk_arguments,
        ),
        toolchain=(
            backend,
            str(backend_sdk.root),
            backend_sdk.version,
            sys.executable,
            str(sysconfig.get_config_var("SOABI")),
            pybind11.__version__,
            pybind11.get_cmake_dir(),
        ),
        toolchain_files=(Path(sys.executable), *backend_sdk_files),
    )

    build.configure()
    return build


def get_backend_module(backend: Backend | str) -> ModuleType:
    backend = Backend(backend)
    if backend == Backend.HIP:
        raise NotImplementedError("hip backend is not implemented yet")

    module_name = f"gpu_libm_bench.backends._{backend}"
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name != module_name:
            raise
    else:
        return _initialize_backend_module(module)

    directory = paths.backend_build_directory(backend) / "python"
    for suffix in EXTENSION_SUFFIXES:
        path = directory / f"_{backend}{suffix}"
        if path.is_file():
            return _initialize_backend_module(
                _load_backend_module(path, module_name=module_name)
            )

    raise BackendModuleNotFoundError(
        f"native backend {module_name!r} was not found in {directory}. "
        f"Build it with 'python -m gpu_libm_bench.compiler backend backend={backend}'."
    )


def _load_backend_module(path: Path, *, module_name: str) -> ModuleType:
    path = path.resolve(strict=True)

    existing = sys.modules.get(module_name)
    if existing is not None:
        if existing.__file__ is not None and Path(existing.__file__).resolve() == path:
            return existing
        raise ValueError(f"backend {module_name!r} is already loaded from another path")

    spec = spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load backend {module_name!r} from {path}")

    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[module_name] = module

    return module


def _initialize_backend_module(module: ModuleType) -> ModuleType:
    if not getattr(module, "_images_build_root_configured", False):
        module._set_images_build_root(str(paths.images_build_root()))
        module.__dict__["_images_build_root_configured"] = True
    return module
