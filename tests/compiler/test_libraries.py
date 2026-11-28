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

"""Test math library resolution for compilation targets."""

from pathlib import Path

import pytest

from gpu_libm_bench.compiler import libraries
from gpu_libm_bench.compiler.targets import CompilationTarget
from gpu_libm_bench.compiler.toolchains import CUDAToolkit, LLVMToolchain, ROCmCoreSDK
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

    expected_reflect = ()
    if library == "libdevice":
        if override:
            library_config.nvvm_reflect["cuda-arch"].value = 900
            del library_config.nvvm_reflect["cuda-prec-sqrt"]
            expected_reflect = (
                ("__CUDA_ARCH", 900),
                ("__CUDA_FTZ", 0),
            )
        else:
            expected_reflect = (
                ("__CUDA_ARCH", 1200),
                ("__CUDA_FTZ", 0),
                ("__CUDA_PREC_SQRT", 1),
            )

    def inspect(command: list[str | Path]) -> str:
        if command[0] == llvm.clang:
            assert library == "libdevice" and not override
            assert f"--target={TARGET.target_triple}" in command
            assert f"-march={TARGET.architecture}" in command
            return "#define __CUDA_ARCH__ 1200"
        assert expected_path in command
        if command[0] == llvm.llvm_dis:
            return f'target triple = "{triple}"\n'
        return symbol

    monkeypatch.setattr(libraries, "run_tool", inspect)

    result = libraries.resolve_library(
        library_config=library_config, llvm=llvm, target=TARGET, backend_sdk=backend_sdk
    )

    assert result == libraries.ResolvedLibrary(
        library, expected_path, (("expf", symbol),), expected_reflect
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


def test_nvptx_rejects_out_of_range_reflect_value(
    llvm: LLVMToolchain, backend_sdk: CUDAToolkit
) -> None:
    library = load_library_config("libdevice")
    library.nvvm_reflect["cuda-arch"].value = 2**32

    with pytest.raises(ValueError, match="unsigned 32-bit integer"):
        libraries.resolve_library(
            library_config=library, llvm=llvm, target=TARGET, backend_sdk=backend_sdk
        )


@pytest.mark.parametrize("name", ["cuda-ftz", "cuda-prec-sqrt"])
def test_libdevice_requires_explicit_reflect_value(
    llvm: LLVMToolchain,
    backend_sdk: CUDAToolkit,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    library = load_library_config("libdevice")
    entry = library.nvvm_reflect[name]
    entry.value = None
    library.nvvm_reflect = {name: entry}
    monkeypatch.setattr(libraries, "_validate_bitcode", lambda *args, **kwargs: None)

    with pytest.raises(ValueError, match=f"{entry.symbol}.*provide an explicit value"):
        libraries.resolve_library(
            library_config=library, llvm=llvm, target=TARGET, backend_sdk=backend_sdk
        )


def test_amdgcn_ignores_nvvm_reflect(
    llvm: LLVMToolchain, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = CompilationTarget("hip", "amdgcn-amd-amdhsa", "gfx942")
    library = load_library_config("llvm-libm")
    library.backends["hip"] = library.backends["cuda"]
    library.nvvm_reflect = load_library_config("libdevice").nvvm_reflect
    library.nvvm_reflect["cuda-ftz"].value = -1

    path = llvm.library_dirs[0] / target.target_triple / "libm.bc"
    path.parent.mkdir(parents=True)
    path.touch()
    monkeypatch.setattr(libraries, "_validate_bitcode", lambda *args, **kwargs: None)

    result = libraries.resolve_library(
        library_config=library,
        llvm=llvm,
        target=target,
        backend_sdk=ROCmCoreSDK(llvm.root, "7.0"),
    )
    assert result.nvvm_reflect == ()
