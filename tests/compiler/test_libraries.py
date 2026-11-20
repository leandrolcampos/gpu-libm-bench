# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Test bitcode discovery, compatibility, and resolved function bindings."""

from pathlib import Path

import pytest

from gpu_libm_bench.compiler import libraries
from gpu_libm_bench.compiler.targets import CompilationTarget
from gpu_libm_bench.compiler.toolchains import LLVMToolchain
from gpu_libm_bench.configs import (
    LibraryConfig,
    LibraryFunctionConfig,
    load_library_config,
)

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

    expected_path = default
    symbol = "expf"
    if override:
        expected_path = tmp_path / "custom.bc"
        expected_path.touch()
        library_config.backends["cuda"].bitcode_path = str(expected_path)
        symbol = "my_expf"
        library_config.functions["expf"].symbol = symbol

    def inspect(command: list[str | Path]) -> str:
        assert expected_path in command
        if command[0] == llvm.llvm_dis:
            return f'target triple = "{TARGET.target_triple}"\n'
        return symbol

    monkeypatch.setattr(libraries, "run_tool", inspect)

    result = libraries.resolve_library(
        library_config=library_config, llvm=llvm, target=TARGET
    )

    assert result.bitcode_path == expected_path
    assert result.name == "llvm-libm"
    assert result.bindings == (("expf", symbol),)


@pytest.mark.parametrize(
    ("name", "error", "message"),
    [
        ("unknown", ValueError, "unsupported library"),
        ("llvm-libm", libraries.LibraryUnavailableError, "support for backend 'cuda'"),
    ],
)
def test_library_must_support_target(
    llvm: LLVMToolchain, name: str, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        libraries.resolve_library(
            library_config=LibraryConfig(name, {}), llvm=llvm, target=TARGET
        )


def test_missing_explicit_bitcode_does_not_fall_back(
    llvm: LLVMToolchain, library_config: LibraryConfig
) -> None:
    default = llvm.library_dirs[0] / TARGET.target_triple / "libm.bc"
    default.parent.mkdir(parents=True)
    default.touch()
    library_config.backends["cuda"].bitcode_path = str(llvm.root / "missing.bc")

    with pytest.raises(libraries.LibraryUnavailableError, match="missing.bc"):
        libraries.resolve_library(
            library_config=library_config, llvm=llvm, target=TARGET
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
    library_config.backends["cuda"].bitcode_path = str(path)

    monkeypatch.setattr(
        libraries,
        "run_tool",
        lambda command: f'target triple = "{triple}"' if triple else "",
    )

    with pytest.raises(error, match="target triple"):
        libraries.resolve_library(
            library_config=library_config, llvm=llvm, target=TARGET
        )


def test_bitcode_must_export_all_configured_symbols(
    monkeypatch: pytest.MonkeyPatch,
    llvm: LLVMToolchain,
    library_config: LibraryConfig,
) -> None:
    path = llvm.root / "libm.bc"
    path.touch()
    library_config.backends["cuda"].bitcode_path = str(path)
    library_config.functions["logf"] = LibraryFunctionConfig(symbol="my_logf")

    monkeypatch.setattr(
        libraries,
        "run_tool",
        lambda command: (
            f'target triple = "{TARGET.target_triple}"'
            if command[0] == llvm.llvm_dis
            else "expf\nlogf\n"
        ),
    )

    with pytest.raises(
        libraries.LibraryUnavailableError, match="bindings: logf -> my_logf"
    ):
        libraries.resolve_library(
            library_config=library_config, llvm=llvm, target=TARGET
        )
