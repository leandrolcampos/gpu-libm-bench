# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Tests for math library resolution and compatibility."""

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
    default = llvm.library_dirs[0] / TARGET.target_triple / "libm.bc"
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

    result = libraries.resolve_library(
        library="llvm-libm",
        llvm=llvm,
        target=TARGET,
        bitcode_path=custom if override else None,
        required_symbols=("expf",),
    )

    assert result.bitcode_path == selected
    assert result.library == "llvm-libm"


def test_support_library_needs_no_external_bitcode(
    monkeypatch: pytest.MonkeyPatch, llvm: LLVMToolchain
) -> None:
    def unexpected_command(*args, **kwargs):
        pytest.fail("resolving support must not inspect external bitcode")

    monkeypatch.setattr(libraries, "run_tool", unexpected_command)
    resolved_library = libraries.resolve_library(
        library="support", llvm=llvm, target=TARGET
    )

    assert resolved_library == libraries.ResolvedLibrary("support")


def test_unsupported_library_is_rejected(llvm: LLVMToolchain) -> None:
    with pytest.raises(ValueError, match="unsupported library"):
        libraries.resolve_library(library="unknown", llvm=llvm, target=TARGET)


def test_missing_explicit_bitcode_does_not_fall_back(llvm: LLVMToolchain) -> None:
    default = llvm.library_dirs[0] / TARGET.target_triple / "libm.bc"
    default.parent.mkdir(parents=True)
    default.touch()

    with pytest.raises(libraries.LibraryUnavailableError, match="missing.bc"):
        libraries.resolve_library(
            library="llvm-libm",
            llvm=llvm,
            target=TARGET,
            bitcode_path=llvm.root / "missing.bc",
        )


@pytest.mark.parametrize(
    ("triple", "error"),
    [("amdgcn-amd-amdhsa", libraries.LibraryUnavailableError), (None, ValueError)],
)
def test_bitcode_requires_matching_target(
    monkeypatch: pytest.MonkeyPatch,
    llvm: LLVMToolchain,
    triple: str | None,
    error: type[Exception],
) -> None:
    path = llvm.root / "libm.bc"
    path.touch()
    monkeypatch.setattr(
        libraries,
        "run_tool",
        lambda command: f'target triple = "{triple}"' if triple else "",
    )

    with pytest.raises(error, match="target triple"):
        libraries.resolve_library(
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

    with pytest.raises(
        libraries.LibraryUnavailableError, match="does not define symbols: expf"
    ):
        libraries.resolve_library(
            library="llvm-libm",
            llvm=llvm,
            target=TARGET,
            bitcode_path=path,
            required_symbols=("expf",),
        )
