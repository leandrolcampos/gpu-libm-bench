# Copyright 2026 Leandro A. Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Resolution of compiler toolchains and GPU SDKs."""

import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from ._commands import executable, run_tool


@dataclass(frozen=True, slots=True)
class LLVMToolchain:
    root: Path
    clang: Path
    llvm_dis: Path
    llvm_nm: Path
    version: str


@dataclass(frozen=True, slots=True)
class CudaToolkit:
    root: Path
    ptxas: Path
    version: str


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
                    "or add clang++ to PATH."
                )
            root = Path(discovered).resolve().parent.parent

    root = root.expanduser().resolve()
    clang = executable(root / "bin" / "clang++")
    llvm_dis = executable(root / "bin" / "llvm-dis")
    llvm_nm = executable(root / "bin" / "llvm-nm")

    versions = [run_tool([tool, "--version"]) for tool in (clang, llvm_dis, llvm_nm)]
    majors = []
    for tool, version in zip((clang, llvm_dis, llvm_nm), versions, strict=True):
        match = re.search(r"\b(?:clang|LLVM) version\s+(\d+)", version)
        if match is None:
            raise ValueError(f"Cannot identify the LLVM version of {tool}: {version}")
        majors.append(match.group(1))
    if len(set(majors)) != 1:
        raise ValueError(f"LLVM tools in {root} have incompatible versions: {majors}")

    return LLVMToolchain(root, clang, llvm_dis, llvm_nm, versions[0])


def resolve_cuda(root: Path | None = None) -> CudaToolkit:
    cmake = shutil.which("cmake")
    if cmake is None:
        raise FileNotFoundError(
            "CMake was not found on PATH; it is required to discover the CUDA Toolkit."
        )

    probe = Path(__file__).resolve().parents[3] / "cmake" / "probes" / "cuda"
    if not (probe / "CMakeLists.txt").is_file():
        raise FileNotFoundError(
            f"CUDA Toolkit discovery project was not found: {probe}"
        )

    if root is not None:
        root = root.expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"CUDA SDK directory does not exist: {root}")

    with TemporaryDirectory(prefix="gpu-libm-bench-cuda-") as directory:
        command: list[str | Path] = [cmake, "-S", probe, "-B", directory, "-G", "Ninja"]
        if root is not None:
            command.append(f"-DCUDAToolkit_ROOT={root}")
        run_tool(command)
        metadata = json.loads((Path(directory) / "cuda-toolkit.json").read_text())
    selected = Path(metadata["root"]).resolve()
    if root is not None and selected != root:
        raise ValueError(
            f"CUDA discovery selected {selected} instead of requested {root}."
        )

    ptxas = executable(Path(metadata["ptxas"]).resolve())
    version = run_tool([ptxas, "--version"])
    if "ptxas" not in version:
        raise ValueError(f"Cannot identify the CUDA assembler {ptxas}: {version}")

    return CudaToolkit(selected, ptxas, metadata["version"])
