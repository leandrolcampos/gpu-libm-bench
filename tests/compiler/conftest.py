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
