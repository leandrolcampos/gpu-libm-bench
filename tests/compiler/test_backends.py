# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Tests for native backend build paths, discovery, and backend SDK compatibility."""

from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest

from gpu_libm_bench.compiler import backends
from gpu_libm_bench.compiler._cmake import CMakeBuild
from gpu_libm_bench.compiler.backends import configure_backend
from gpu_libm_bench.compiler.toolchains import BackendSDK, CUDAToolkit, ROCmCoreSDK


@pytest.mark.parametrize(
    ("build_directory", "expected_directory"),
    [
        (None, "build/backends/cuda"),
        ("custom", "custom"),
    ],
)
def test_backend_build_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    build_directory: str | None,
    expected_directory: str,
) -> None:
    pybind11 = SimpleNamespace(__version__="3.0", get_cmake_dir=lambda: "/pybind")
    monkeypatch.setattr(backends, "import_module", lambda _: pybind11)
    monkeypatch.setattr(CMakeBuild, "configure", lambda self: None)
    monkeypatch.setattr(backends.paths, "build_root", lambda: tmp_path / "build")

    backend_sdk = CUDAToolkit(Path("/cuda"), Path("/cuda/bin/ptxas"), "13.4")
    build = configure_backend(
        "cuda",
        backend_sdk=backend_sdk,
        build_directory=tmp_path / build_directory
        if build_directory is not None
        else None,
    )

    expected = tmp_path / expected_directory
    assert build.source_directory == backends.paths.source_root() / "backends/cuda"
    assert build.build_directory == expected


@pytest.mark.parametrize(
    ("backend", "backend_sdk", "error", "message"),
    [
        ("cuda", object(), TypeError, "unsupported backend SDK type: object"),
        (
            "hip",
            CUDAToolkit(Path("/cuda"), Path("/cuda/bin/ptxas"), "13.4"),
            ValueError,
            "non-cuda backend",
        ),
        (
            "cuda",
            ROCmCoreSDK(Path("/rocm"), "test"),
            ValueError,
            "non-hip backend",
        ),
        (
            "hip",
            ROCmCoreSDK(Path("/rocm"), "test"),
            NotImplementedError,
            "hip backend.*not implemented",
        ),
    ],
)
def test_backend_configuration_rejects_incompatible_backend_sdk(
    backend: str, backend_sdk: object, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        configure_backend(backend, backend_sdk=cast(BackendSDK, backend_sdk))


def test_backend_initialization_retries_failures_and_reuses_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = ModuleType("gpu_libm_bench.backends._cuda")
    initialize = Mock(side_effect=[RuntimeError("initialization failed"), None])
    module._set_images_build_root = initialize
    monkeypatch.setattr(backends, "import_module", lambda _: module)
    monkeypatch.setattr(backends.paths, "build_root", lambda: tmp_path / "build")

    with pytest.raises(RuntimeError, match="initialization failed"):
        backends.get_backend_module("cuda")

    assert backends.get_backend_module("cuda") is module
    assert backends.get_backend_module("cuda") is module
    assert initialize.call_count == 2
    initialize.assert_called_with(str(tmp_path / "build/kernels"))


@pytest.mark.parametrize(
    ("backend", "error"), [("hip", NotImplementedError), ("unknown", ValueError)]
)
def test_backend_module_selection_rejects_unavailable_backends(backend, error) -> None:
    with pytest.raises(error):
        backends.get_backend_module(backend)
