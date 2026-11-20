# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Test bitcode discovery, compatibility, and resolved function bindings."""

from pathlib import Path

import pytest

from gpu_libm_bench.compiler import libraries
from gpu_libm_bench.compiler.targets import CompilationTarget
from gpu_libm_bench.compiler.toolchains import LLVMToolchain
from gpu_libm_bench.configs import LibraryConfig, load_library_config

TARGET = CompilationTarget("cuda", "nvptx64-nvidia-cuda", "sm_120")


@pytest.fixture
def library_config() -> LibraryConfig:
    return load_library_config("llvm-libm")


@pytest.mark.parametrize("override", [False, True])
def test_library_resolves_bitcode_and_bindings(
    monkeypatch: pytest.MonkeyPatch,
    llvm: LLVMToolchain,
    tmp_path: Path,
    library_config: LibraryConfig,
    override: bool,
) -> None:
    default = llvm.library_dirs[0] / TARGET.target_triple / "libm.bc"
    default.parent.mkdir(parents=True)
    default.touch()

    bitcode_path = None
    symbol = "expf"
    if override:
        bitcode_path = tmp_path / "custom.bc"
        bitcode_path.touch()
        symbol = "my_expf"
        library_config.functions["expf"].symbol = symbol

    expected_path = bitcode_path if bitcode_path is not None else default

    def inspect(command: list[str | Path]) -> str:
        assert expected_path in command
        if command[0] == llvm.llvm_dis:
            return f'target triple = "{TARGET.target_triple}"\n'
        return symbol

    monkeypatch.setattr(libraries, "run_tool", inspect)

    result = libraries.resolve_library(
        library_config=library_config,
        llvm=llvm,
        target=TARGET,
        bitcode_path=bitcode_path,
        required_symbols=(symbol,),
    )

    assert result.bitcode_path == expected_path
    assert result.name == "llvm-libm"
    assert result.bindings == (("expf", symbol),)


def test_unsupported_library_is_rejected(llvm: LLVMToolchain) -> None:
    with pytest.raises(ValueError, match="unsupported library"):
        libraries.resolve_library(
            library_config=LibraryConfig("unknown", {}), llvm=llvm, target=TARGET
        )


def test_missing_explicit_bitcode_does_not_fall_back(
    llvm: LLVMToolchain, library_config: LibraryConfig
) -> None:
    default = llvm.library_dirs[0] / TARGET.target_triple / "libm.bc"
    default.parent.mkdir(parents=True)
    default.touch()

    with pytest.raises(libraries.LibraryUnavailableError, match="missing.bc"):
        libraries.resolve_library(
            library_config=library_config,
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
    library_config: LibraryConfig,
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
            library_config=library_config, llvm=llvm, target=TARGET, bitcode_path=path
        )


def test_bitcode_must_define_required_function(
    monkeypatch: pytest.MonkeyPatch,
    llvm: LLVMToolchain,
    library_config: LibraryConfig,
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
            library_config=library_config,
            llvm=llvm,
            target=TARGET,
            bitcode_path=path,
            required_symbols=("expf",),
        )
