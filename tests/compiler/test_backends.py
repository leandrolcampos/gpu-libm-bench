# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Tests for native backend build paths, discovery, and SDK compatibility."""

from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest

from gpu_libm_bench.compiler import backends
from gpu_libm_bench.compiler._cmake import CMakeBuild
from gpu_libm_bench.compiler.backends import configure_backend
from gpu_libm_bench.compiler.toolchains import CUDAToolkit


@pytest.mark.parametrize(
    ("backend", "sdk", "error", "message"),
    [
        ("cuda", object(), TypeError, "unsupported SDK type: object"),
        (
            "rocm",
            CUDAToolkit(Path("/cuda"), Path("/cuda/bin/ptxas"), "13.4"),
            ValueError,
            "non-CUDA backend",
        ),
    ],
)
def test_backend_configuration_rejects_incompatible_sdk(
    backend: str, sdk: object, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        configure_backend(backend, sdk=cast(CUDAToolkit, sdk))


@pytest.mark.parametrize(
    ("backend_root", "build_directory", "expected_directory"),
    [
        (None, None, "build/backends/cuda"),
        ("native", None, "native/cuda"),
        ("native", "custom", "custom"),
    ],
)
def test_backend_build_directory_precedence_and_discovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend_root: str | None,
    build_directory: str | None,
    expected_directory: str,
) -> None:
    def import_module(name: str):
        if name == "pybind11":
            return SimpleNamespace(__version__="3.0", get_cmake_dir=lambda: "/pybind")
        raise ModuleNotFoundError(name=name)

    monkeypatch.setattr(backends, "import_module", import_module)
    monkeypatch.setattr(CMakeBuild, "configure", lambda self: None)
    monkeypatch.setattr(backends._paths, "build_root", lambda: tmp_path / "build")
    if backend_root is None:
        monkeypatch.delenv("GPU_LIBM_BENCH_BACKEND_DIR", raising=False)
    else:
        monkeypatch.setenv("GPU_LIBM_BENCH_BACKEND_DIR", str(tmp_path / backend_root))

    sdk = CUDAToolkit(Path("/cuda"), Path("/cuda/bin/ptxas"), "13.4")

    build = configure_backend(
        "cuda",
        sdk=sdk,
        build_directory=tmp_path / build_directory
        if build_directory is not None
        else None,
    )

    expected = tmp_path / expected_directory
    assert build.source_directory == backends._paths.source_root() / "backends/cuda"
    assert build.build_directory == expected

    if build_directory is None:
        path = expected / "python" / f"_cuda{backends.EXTENSION_SUFFIXES[0]}"
        path.parent.mkdir(parents=True)
        path.touch()

        module = ModuleType("gpu_libm_bench.backends._cuda")
        loader = Mock(return_value=module)
        monkeypatch.setattr(backends, "_load_backend", loader)

        assert backends.get_backend("cuda") is module
        loader.assert_called_once_with(path, module_name=module.__name__)
