# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Test bitcode discovery, compatibility, and resolved function bindings."""

from pathlib import Path

import pytest

from gpu_libm_bench.compiler import libraries
from gpu_libm_bench.compiler.targets import CompilationTarget
from gpu_libm_bench.compiler.toolchains import CUDAToolkit, LLVMToolchain
from gpu_libm_bench.configs import (
    LibraryConfig,
    LibraryFunctionConfig,
    load_library_config,
)

TARGET = CompilationTarget("cuda", "nvptx64-nvidia-cuda", "sm_120")
LIBDEVICE_TRIPLE = "nvptx64-nvidia-gpulibs"


@pytest.fixture
def backend_sdk(tmp_path: Path) -> CUDAToolkit:
    root = tmp_path / "cuda"
    return CUDAToolkit(root, root / "bin/ptxas", "13.4")


@pytest.fixture
def library_config() -> LibraryConfig:
    return load_library_config("llvm-libm")


@pytest.mark.parametrize(
    ("name", "error", "message"),
    [
        ("unknown", ValueError, "unsupported library"),
        ("llvm-libm", libraries.LibraryUnavailableError, "support for backend 'cuda'"),
    ],
)
def test_library_must_be_known_and_support_backend(
    llvm: LLVMToolchain,
    backend_sdk: CUDAToolkit,
    name: str,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        libraries.resolve_library(
            library_config=LibraryConfig(name, {}),
            llvm=llvm,
            target=TARGET,
            backend_sdk=backend_sdk,
        )


@pytest.mark.parametrize(
    ("library", "triple", "symbol", "override"),
    [
        ("llvm-libm", TARGET.target_triple, "expf", False),
        ("libdevice", LIBDEVICE_TRIPLE, "__nv_expf", False),
        ("libdevice", LIBDEVICE_TRIPLE, "__nv_expf", True),
    ],
)
def test_library_resolves_bitcode_and_bindings(
    monkeypatch: pytest.MonkeyPatch,
    llvm: LLVMToolchain,
    tmp_path: Path,
    backend_sdk: CUDAToolkit,
    library: str,
    triple: str,
    symbol: str,
    override: bool,
) -> None:
    library_config = load_library_config(library)
    default = (
        backend_sdk.root / "nvvm/libdevice/libdevice.10.bc"
        if library == "libdevice"
        else llvm.library_dirs[0] / TARGET.target_triple / "libm.bc"
    )
    default.parent.mkdir(parents=True)
    default.touch()

    expected_path = default
    if override:
        expected_path = tmp_path / "custom.bc"
        expected_path.touch()
        library_config.backends["cuda"].bitcode_path = str(expected_path)

    def inspect(command: list[str | Path]) -> str:
        assert expected_path in command
        if command[0] == llvm.llvm_dis:
            return f'target triple = "{triple}"\n'
        return symbol

    monkeypatch.setattr(libraries, "run_tool", inspect)

    result = libraries.resolve_library(
        library_config=library_config, llvm=llvm, target=TARGET, backend_sdk=backend_sdk
    )

    assert result == libraries.ResolvedLibrary(
        library, expected_path, (("expf", symbol),)
    )


def test_missing_explicit_bitcode_does_not_fall_back(
    llvm: LLVMToolchain, backend_sdk: CUDAToolkit, library_config: LibraryConfig
) -> None:
    default = llvm.library_dirs[0] / TARGET.target_triple / "libm.bc"
    default.parent.mkdir(parents=True)
    default.touch()
    library_config.backends["cuda"].bitcode_path = str(llvm.root / "missing.bc")

    with pytest.raises(libraries.LibraryUnavailableError, match="missing.bc"):
        libraries.resolve_library(
            library_config=library_config,
            llvm=llvm,
            target=TARGET,
            backend_sdk=backend_sdk,
        )


@pytest.mark.parametrize(
    ("library", "triple", "error"),
    [
        ("llvm-libm", LIBDEVICE_TRIPLE, libraries.LibraryUnavailableError),
        ("llvm-libm", None, ValueError),
        ("libdevice", "amdgcn-amd-amdhsa", libraries.LibraryUnavailableError),
    ],
)
def test_bitcode_requires_matching_target(
    monkeypatch: pytest.MonkeyPatch,
    llvm: LLVMToolchain,
    backend_sdk: CUDAToolkit,
    library: str,
    triple: str | None,
    error: type[Exception],
) -> None:
    path = llvm.root / "libm.bc"
    path.touch()
    library_config = load_library_config(library)
    library_config.backends["cuda"].bitcode_path = str(path)

    monkeypatch.setattr(
        libraries,
        "run_tool",
        lambda command: f'target triple = "{triple}"' if triple else "",
    )

    with pytest.raises(error, match="target triple"):
        libraries.resolve_library(
            library_config=library_config,
            llvm=llvm,
            target=TARGET,
            backend_sdk=backend_sdk,
        )


def test_bitcode_must_export_all_configured_symbols(
    monkeypatch: pytest.MonkeyPatch,
    llvm: LLVMToolchain,
    backend_sdk: CUDAToolkit,
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
            library_config=library_config,
            llvm=llvm,
            target=TARGET,
            backend_sdk=backend_sdk,
        )
