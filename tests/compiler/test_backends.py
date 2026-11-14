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

    sdk = CUDAToolkit(Path("/cuda"), Path("/cuda/bin/ptxas"), "13.4")
    build = configure_backend(
        "cuda",
        sdk=sdk,
        build_directory=tmp_path / build_directory
        if build_directory is not None
        else None,
    )

    expected = tmp_path / expected_directory
    assert build.source_directory == backends.paths.source_root() / "backends/cuda"
    assert build.build_directory == expected


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


def test_backend_initialization_retries_failures_and_reuses_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = ModuleType("gpu_libm_bench.backends._cuda")
    initialize = Mock(side_effect=[RuntimeError("initialization failed"), None])
    module._set_images_build_root = initialize
    monkeypatch.setattr(backends, "import_module", lambda _: module)
    monkeypatch.setattr(backends.paths, "build_root", lambda: tmp_path / "build")

    with pytest.raises(RuntimeError, match="initialization failed"):
        backends.get_backend("cuda")

    assert backends.get_backend("cuda") is module
    assert backends.get_backend("cuda") is module
    assert initialize.call_count == 2
    initialize.assert_called_with(str(tmp_path / "build/kernels"))
