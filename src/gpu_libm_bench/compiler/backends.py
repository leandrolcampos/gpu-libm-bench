# Copyright 2026 Leandro A. Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Native backend compilation and loading."""

import os
import sys
from importlib import import_module
from importlib.machinery import EXTENSION_SUFFIXES
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType


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
            raise ImportError(
                f"Native backend {module_name!r} is not installed. "
                "Set GPU_LIBM_BENCH_BACKEND_DIR to the native backend build directory."
            )
        backend_directory = package_directory.parent.parent / "build" / "backends"

    directory = backend_directory / backend / "python"
    for suffix in EXTENSION_SUFFIXES:
        path = directory / f"_{backend}{suffix}"
        if path.is_file():
            return load_backend(path, module_name=module_name)
    raise ImportError(
        f"Native backend {module_name!r} was not found in {directory}. "
        f"Build it with 'pixi run -e native {backend}-build', or set "
        "GPU_LIBM_BENCH_BACKEND_DIR to the native backend build directory."
    )


def load_backend(path: Path, *, module_name: str) -> ModuleType:
    path = path.resolve(strict=True)
    existing = sys.modules.get(module_name)
    if existing is not None:
        if existing.__file__ is not None and Path(existing.__file__).resolve() == path:
            return existing
        raise ValueError(
            f"Backend {module_name!r} is already loaded from another path."
        )

    spec = spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load backend {module_name!r} from {path}.")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[module_name] = module
    return module
