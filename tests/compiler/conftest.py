# Copyright 2026 Leandro A. Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for compilation dependency resolution."""

from pathlib import Path

import pytest

from gpu_libm_bench.compiler.toolchains import LLVMToolchain


@pytest.fixture
def llvm(tmp_path: Path) -> LLVMToolchain:
    root = tmp_path / "llvm"
    (root / "bin").mkdir(parents=True)
    paths = [root / "bin" / name for name in ("clang++", "llvm-dis", "llvm-nm")]
    for path in paths:
        path.touch()
        path.chmod(0o755)
    return LLVMToolchain(root, *paths, "clang version 24.0.0git")
