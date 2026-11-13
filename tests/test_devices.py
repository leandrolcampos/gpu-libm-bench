# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Tests for GPU device interfaces."""

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from gpu_libm_bench.compiler import backends
from gpu_libm_bench.devices import DeviceContext, LaunchConfig


def test_import_does_not_load_cuda(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import gpu_libm_bench.devices; "
            "assert 'gpu_libm_bench.backends._cuda' not in sys.modules",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_query_discovers_and_reuses_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module_name = "gpu_libm_bench.backends._cuda"
    monkeypatch.setitem(sys.modules, module_name, None)
    monkeypatch.delitem(sys.modules, module_name)
    monkeypatch.setattr(backends.paths, "build_root", lambda: tmp_path)
    monkeypatch.setattr(backends, "EXTENSION_SUFFIXES", [".py"])

    directory = tmp_path / "backends/cuda/python"
    directory.mkdir(parents=True)
    path = directory / "_cuda.py"
    path.write_text(
        "roots = []\n"
        "def _set_images_build_root(root):\n    roots.append(root)\n"
        "def device_count():\n    return 2\n"
    )

    assert DeviceContext.device_count() == 2

    path.unlink()
    assert DeviceContext.device_count() == 2
    module = sys.modules[module_name]
    assert module.roots == [str(backends.paths.images_build_root())]


def test_missing_backend_has_an_actionable_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module_name = "gpu_libm_bench.backends._cuda"

    def missing_module(name: str) -> None:
        raise ModuleNotFoundError(f"No module named {name!r}", name=name)

    monkeypatch.setattr(backends.paths, "build_root", lambda: tmp_path)
    monkeypatch.setattr(backends, "import_module", missing_module)

    with pytest.raises(
        backends.BackendNotFoundError,
        match="python -m gpu_libm_bench.compiler backend backend=cuda",
    ) as error:
        DeviceContext.device_count()

    assert module_name in str(error.value)
    assert str(tmp_path / "backends/cuda/python") in str(error.value)


@pytest.mark.parametrize(
    "error",
    [
        ImportError("libcuda.so.1: cannot open shared object file"),
        ModuleNotFoundError("Missing backend dependency", name="backend_dependency"),
    ],
)
def test_backend_dependency_failure_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
    error: ImportError,
) -> None:
    def broken_module(name: str) -> None:
        raise error

    monkeypatch.setattr(backends, "import_module", broken_module)

    with pytest.raises(RuntimeError) as caught:
        DeviceContext.device_count()

    assert caught.value.__cause__ is error
    assert not isinstance(caught.value, backends.BackendNotFoundError)


@pytest.mark.parametrize(
    ("parameters", "error", "message"),
    [
        ({"grid": (1, 1)}, ValueError, "grid must have exactly three dimensions"),
        (
            {"block": (1, 1, 1, 1)},
            ValueError,
            "block must have exactly three dimensions",
        ),
        ({"grid": (-1, 1, 1)}, ValueError, "grid dimensions must be positive"),
        ({"block": (1, 0, 1)}, ValueError, "block dimensions must be positive"),
        ({"grid": (1, 1.5, 1)}, TypeError, "grid dimensions must be integers"),
        ({"block": (1, "1", 1)}, TypeError, "block dimensions must be integers"),
        (
            {"shared_memory_bytes": -1},
            ValueError,
            "shared_memory_bytes must be nonnegative",
        ),
        (
            {"shared_memory_bytes": 1.5},
            TypeError,
            "shared_memory_bytes must be an integer",
        ),
    ],
)
def test_invalid_launch_configuration(
    parameters: dict[str, Any], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        LaunchConfig(**parameters)
