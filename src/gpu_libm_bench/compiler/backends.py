# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Native backend compilation and loading."""

import os
import sys
import sysconfig
from importlib import import_module
from importlib.machinery import EXTENSION_SUFFIXES
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

from ._cmake import CMakeBuild, project_directory
from .toolchains import CUDAToolkit


class BackendNotFoundError(RuntimeError):
    pass


def build_backend(
    backend: str,
    *,
    sdk: CUDAToolkit,
    build_directory: Path | None = None,
    build_tests: bool = False,
) -> Path:
    build = configure_backend(
        backend, sdk=sdk, build_directory=build_directory, build_tests=build_tests
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
    backend: str,
    *,
    sdk: CUDAToolkit,
    build_directory: Path | None = None,
    build_tests: bool = False,
) -> CMakeBuild:
    if isinstance(sdk, CUDAToolkit):
        if backend != "cuda":
            raise ValueError("a CUDA Toolkit cannot be used for a non-CUDA backend")
        sdk_arguments = (f"-DCUDAToolkit_ROOT={sdk.root}",)
        sdk_files = (sdk.ptxas,)
    else:
        raise TypeError(f"unsupported SDK type: {type(sdk).__name__}")

    pybind11 = import_module("pybind11")
    project = project_directory()

    build = CMakeBuild(
        project / "backends" / backend,
        build_directory
        if build_directory is not None
        else project / "build" / "backends" / backend,
        arguments=(
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DBUILD_TESTING={'ON' if build_tests else 'OFF'}",
            *sdk_arguments,
        ),
        toolchain=(
            backend,
            str(sdk.root),
            sdk.version,
            sys.executable,
            str(sysconfig.get_config_var("SOABI")),
            pybind11.__version__,
            pybind11.get_cmake_dir(),
        ),
        toolchain_files=(Path(sys.executable), *sdk_files),
    )

    build.configure()
    return build


def get_backend(backend: str) -> ModuleType:
    module_name = f"gpu_libm_bench.backends._{backend}"

    try:
        return import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name != module_name:
            raise

    configured_directory = os.environ.get("GPU_LIBM_BENCH_BACKEND_DIR")
    if configured_directory is not None:
        backend_directory = Path(configured_directory).expanduser()
    else:
        package_directory = Path(__file__).resolve().parents[1]
        if package_directory.parent.name != "src":
            raise BackendNotFoundError(
                f"native backend {module_name!r} is not installed. "
                "Set GPU_LIBM_BENCH_BACKEND_DIR to the native backend build directory"
            )
        backend_directory = project_directory() / "build" / "backends"

    directory = backend_directory / backend / "python"
    for suffix in EXTENSION_SUFFIXES:
        path = directory / f"_{backend}{suffix}"
        if path.is_file():
            return _load_backend(path, module_name=module_name)

    raise BackendNotFoundError(
        f"native backend {module_name!r} was not found in {directory}. "
        f"Build it with 'pixi run -e native {backend}-build', or set "
        "GPU_LIBM_BENCH_BACKEND_DIR to the native backend build directory"
    )


def _load_backend(path: Path, *, module_name: str) -> ModuleType:
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
