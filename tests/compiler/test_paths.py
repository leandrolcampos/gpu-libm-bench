# ===----------------------------------------------------------------------=== #
# Copyright 2026 Leandro Lacerda Campos
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Tests for checkout discovery independently of the working directory and Git."""

from pathlib import Path

import pytest

from gpu_libm_bench.compiler import paths


def test_checkout_paths_do_not_depend_on_working_directory_or_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "checkout"
    module = root / "src/gpu_libm_bench/compiler/paths.py"
    module.parent.mkdir(parents=True)
    module.touch()
    (root / "pyproject.toml").touch()
    for directory in ("backends", "cmake", "kernels"):
        (root / directory).mkdir()

    monkeypatch.setattr(paths, "__file__", str(module))
    monkeypatch.chdir(tmp_path)

    assert paths.source_root() == root
    assert paths.package_root() == module.parents[1]
    assert paths.build_root() == root / "build"
    assert paths.backends_build_root() == root / "build/backends"
    assert paths.images_build_root() == root / "build/kernels"
    assert paths.backend_build_directory("cuda") == root / "build/backends/cuda"
    assert (
        paths.image_build_directory("llvm-libm", backend="cuda", architecture="sm_120")
        == root / "build/kernels/cuda/sm_120/llvm-libm"
    )
    assert not (root / "build").exists()


def test_missing_native_sources_have_an_actionable_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = tmp_path / "missing/gpu_libm_bench/compiler/paths.py"
    monkeypatch.setattr(paths, "__file__", str(module))

    with pytest.raises(FileNotFoundError, match="editable project checkout"):
        paths.source_root()
