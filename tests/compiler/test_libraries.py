# Copyright 2026 Leandro A. Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Tests for device library selection and target compatibility."""

from pathlib import Path

import pytest

from gpu_libm_bench.compiler import libraries
from gpu_libm_bench.compiler.targets import CompilationTarget
from gpu_libm_bench.compiler.toolchains import LLVMToolchain

TARGET = CompilationTarget("cuda", "nvptx64-nvidia-cuda", "sm_120")


@pytest.mark.parametrize("override", [False, True])
def test_library_path_selection(
    monkeypatch: pytest.MonkeyPatch, llvm: LLVMToolchain, tmp_path: Path, override: bool
) -> None:
    default = llvm.root / "lib" / TARGET.target_triple / "libm.bc"
    default.parent.mkdir(parents=True)
    default.touch()
    custom = tmp_path / "custom.bc"
    custom.touch()
    selected = custom if override else default

    def inspect(command: list[str | Path]) -> str:
        assert selected in command
        if command[0] == llvm.llvm_dis:
            return f'target triple = "{TARGET.target_triple}"\n'
        return "expf\nlogf"

    monkeypatch.setattr(libraries, "run_tool", inspect)
    result = libraries.resolve_device_library(
        library="llvm-libm",
        llvm=llvm,
        target=TARGET,
        bitcode_path=custom if override else None,
        required_symbols=("expf",),
    )
    assert result.bitcode_file == selected


def test_missing_explicit_bitcode_does_not_fall_back(llvm: LLVMToolchain) -> None:
    default = llvm.root / "lib" / TARGET.target_triple / "libm.bc"
    default.parent.mkdir(parents=True)
    default.touch()
    with pytest.raises(FileNotFoundError, match="missing.bc"):
        libraries.resolve_device_library(
            library="llvm-libm",
            llvm=llvm,
            target=TARGET,
            bitcode_path=llvm.root / "missing.bc",
        )


@pytest.mark.parametrize("triple", ["amdgcn-amd-amdhsa", None])
def test_bitcode_requires_matching_target(
    monkeypatch: pytest.MonkeyPatch, llvm: LLVMToolchain, triple: str | None
) -> None:
    path = llvm.root / "libm.bc"
    path.touch()
    monkeypatch.setattr(
        libraries,
        "run_tool",
        lambda command: f'target triple = "{triple}"' if triple else "",
    )
    with pytest.raises(ValueError, match="target triple"):
        libraries.resolve_device_library(
            library="llvm-libm", llvm=llvm, target=TARGET, bitcode_path=path
        )


def test_bitcode_must_define_required_function(
    monkeypatch: pytest.MonkeyPatch, llvm: LLVMToolchain
) -> None:
    path = llvm.root / "libm.bc"
    path.touch()
    monkeypatch.setattr(
        libraries,
        "run_tool",
        lambda command: (
            f'target triple = "{TARGET.target_triple}"'
            if command[0] == llvm.llvm_dis
            else "logf"
        ),
    )
    with pytest.raises(ValueError, match="does not define symbols: expf"):
        libraries.resolve_device_library(
            library="llvm-libm",
            llvm=llvm,
            target=TARGET,
            bitcode_path=path,
            required_symbols=("expf",),
        )


@pytest.mark.parametrize("library", ["cuda-math", "hip-math"])
def test_other_providers_do_not_fall_back_to_llvm_libm(
    llvm: LLVMToolchain, library: str
) -> None:
    with pytest.raises(ValueError, match="Unsupported device math library"):
        libraries.resolve_device_library(library=library, llvm=llvm, target=TARGET)
