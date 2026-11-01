# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for compilation dependency resolution."""

from pathlib import Path

import pytest

from gpu_libm_bench.compiler.toolchains import LLVMToolchain


@pytest.fixture
def llvm(tmp_path: Path) -> LLVMToolchain:
    root = tmp_path / "llvm"
    tools = root / "tools"
    tools.mkdir(parents=True)
    paths = [tools / name for name in ("clang++", "llvm-dis", "llvm-nm")]
    for path in paths:
        path.touch()
        path.chmod(0o755)
    return LLVMToolchain(root, *paths, (root / "lib64",), "clang version 24.0.0git")
