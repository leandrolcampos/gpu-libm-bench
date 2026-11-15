# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Resolution of compiler toolchains and backend SDKs."""

import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from gpu_libm_bench.backends import Backend

from . import paths
from ._commands import executable, run_tool


@dataclass(frozen=True, slots=True)
class LLVMToolchain:
    root: Path
    clang: Path
    llvm_dis: Path
    llvm_nm: Path
    library_dirs: tuple[Path, ...]
    version: str


@dataclass(frozen=True, slots=True)
class CUDAToolkit:
    root: Path
    ptxas: Path
    version: str


@dataclass(frozen=True, slots=True)
class ROCmCoreSDK:
    root: Path
    version: str


type BackendSDK = CUDAToolkit | ROCmCoreSDK


class BackendSDKUnavailableError(RuntimeError):
    pass


def resolve_llvm(root: Path | None = None) -> LLVMToolchain:
    if root is None:
        configured = os.environ.get("LLVM_ROOT")
        if configured is not None:
            root = Path(configured)
        else:
            discovered = shutil.which("clang++")
            if discovered is None:
                raise FileNotFoundError(
                    "LLVM was not found. Set llvm.root or LLVM_ROOT, "
                    "or add clang++ to PATH"
                )
            root = Path(discovered).resolve().parent.parent

    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"LLVM directory does not exist: {root}")

    cmake = shutil.which("cmake")
    if cmake is None:
        raise FileNotFoundError(
            "CMake was not found on PATH; it is required to discover LLVM"
        )
    probe = paths.source_root() / "cmake" / "probes" / "llvm"
    if not (probe / "CMakeLists.txt").is_file():
        raise FileNotFoundError(f"LLVM discovery project was not found: {probe}")
    with TemporaryDirectory(prefix="gpu-libm-bench-llvm-") as directory:
        run_tool(
            [cmake, "-S", probe, "-B", directory, "-G", "Ninja", f"-DLLVM_ROOT={root}"]
        )
        metadata = json.loads((Path(directory) / "llvm-toolchain.json").read_text())

    clang = executable(Path(metadata["clang"]))
    llvm_dis = executable(Path(metadata["llvm_dis"]))
    llvm_nm = executable(Path(metadata["llvm_nm"]))
    library_dirs = tuple(Path(path).resolve() for path in metadata["library_dirs"])

    tools = (clang, llvm_dis, llvm_nm)
    versions = [run_tool([tool, "--version"]) for tool in tools]
    majors = []
    for tool, version in zip(tools, versions, strict=True):
        match = re.search(r"\b(?:clang|LLVM) version\s+(\d+)", version)
        if match is None:
            raise ValueError(f"cannot identify the LLVM version of {tool}: {version}")
        majors.append(match.group(1))
    if len(set(majors)) != 1:
        raise ValueError(f"LLVM tools in {root} have incompatible versions: {majors}")

    return LLVMToolchain(root, clang, llvm_dis, llvm_nm, library_dirs, versions[0])


def resolve_backend_sdk(
    *, backend: Backend | str, root: Path | None = None
) -> BackendSDK:
    backend = Backend(backend)

    match backend:
        case Backend.CUDA:
            return _resolve_cuda(root)
        case Backend.HIP:
            raise NotImplementedError("hip backend is not implemented yet")


def validate_backend_sdk_compatibility(
    backend: Backend | str, backend_sdk: BackendSDK
) -> None:
    backend = Backend(backend)

    match backend_sdk:
        case CUDAToolkit():
            if backend != Backend.CUDA:
                raise ValueError("a CUDA Toolkit cannot be used for a non-cuda backend")
        case ROCmCoreSDK():
            if backend != Backend.HIP:
                raise ValueError("a ROCm Core SDK cannot be used for a non-hip backend")
        case _:
            raise TypeError(
                f"unsupported backend SDK type: {type(backend_sdk).__name__}"
            )


def _resolve_cuda(root: Path | None = None) -> CUDAToolkit:
    cmake = shutil.which("cmake")
    if cmake is None:
        raise FileNotFoundError(
            "CMake was not found on PATH; it is required to discover the CUDA Toolkit"
        )

    probe = paths.source_root() / "cmake" / "probes" / "cuda"
    if not (probe / "CMakeLists.txt").is_file():
        raise FileNotFoundError(
            f"CUDA Toolkit discovery project was not found: {probe}"
        )

    if root is not None:
        root = root.expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"CUDA Toolkit directory does not exist: {root}")

    with TemporaryDirectory(prefix="gpu-libm-bench-cuda-") as directory:
        command: list[str | Path] = [cmake, "-S", probe, "-B", directory, "-G", "Ninja"]
        if root is not None:
            command.append(f"-DCUDAToolkit_ROOT={root}")
        run_tool(command)
        metadata = json.loads((Path(directory) / "cuda-toolkit.json").read_text())

    if not metadata["available"]:
        raise BackendSDKUnavailableError("CUDA Toolkit was not found")

    selected = Path(metadata["root"]).resolve()
    if root is not None and selected != root:
        raise ValueError(
            f"CUDA discovery selected {selected} instead of requested {root}"
        )

    ptxas = executable(Path(metadata["ptxas"]).resolve())
    version = run_tool([ptxas, "--version"])
    if "ptxas" not in version:
        raise ValueError(f"cannot identify the CUDA assembler {ptxas}: {version}")

    return CUDAToolkit(selected, ptxas, metadata["version"])
