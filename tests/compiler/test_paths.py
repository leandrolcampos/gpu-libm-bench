# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Tests for checkout discovery independently of the working directory and Git."""

from pathlib import Path

import pytest

from gpu_libm_bench.compiler import _paths


def test_checkout_paths_do_not_depend_on_working_directory_or_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "checkout"
    module = root / "src/gpu_libm_bench/compiler/_paths.py"
    module.parent.mkdir(parents=True)
    module.touch()
    (root / "pyproject.toml").touch()
    for directory in ("backends", "cmake", "kernels"):
        (root / directory).mkdir()

    monkeypatch.setattr(_paths, "__file__", str(module))
    monkeypatch.chdir(tmp_path)

    assert _paths.source_root() == root
    assert _paths.build_root() == root / "build"
    assert not (root / "build").exists()


def test_missing_native_sources_have_an_actionable_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = tmp_path / "site-packages/gpu_libm_bench/compiler/_paths.py"
    monkeypatch.setattr(_paths, "__file__", str(module))

    with pytest.raises(FileNotFoundError, match="editable project checkout"):
        _paths.source_root()
