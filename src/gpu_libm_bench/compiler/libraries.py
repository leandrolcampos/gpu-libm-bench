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

"""Resolve math libraries and validate the bitcode required by their bindings."""

import re
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from gpu_libm_bench.configs import LibraryConfig

from ._commands import run_tool
from .targets import CompilationTarget
from .toolchains import BackendSDK, CUDAToolkit, LLVMToolchain


@dataclass(frozen=True, slots=True)
class ResolvedLibrary:
    name: str
    bitcode_path: Path | None = None
    bindings: tuple[tuple[str, str], ...] = ()


class LibraryUnavailableError(RuntimeError):
    pass


def resolve_library(
    *,
    library_config: LibraryConfig,
    llvm: LLVMToolchain,
    target: CompilationTarget,
    backend_sdk: BackendSDK,
) -> ResolvedLibrary:
    bitcode_finders = {
        "llvm-libm": partial(_find_llvm_libm_bitcode, llvm, target),
        "libdevice": partial(_find_libdevice_bitcode, backend_sdk),
    }
    name = library_config.name
    find_bitcode = bitcode_finders.get(name)
    if find_bitcode is None:
        raise ValueError(f"unsupported library: {name!r}")

    bitcode_config = library_config.backends.get(target.backend)
    if bitcode_config is None:
        raise LibraryUnavailableError(
            f"library {name!r} does not declare support for "
            f"backend {target.backend.value!r}"
        )

    bitcode_path = bitcode_config.bitcode_path
    path = find_bitcode() if bitcode_path is None else Path(bitcode_path)
    path = path.expanduser().resolve()

    bindings = tuple(
        (function, library_config.symbol_for(function))
        for function in sorted(library_config.functions)
    )

    accepted_triples: tuple[str, ...] = (target.target_triple,)
    if name == "libdevice" and target.target_triple == "nvptx64-nvidia-cuda":
        # libdevice bitcode uses nvptx64-nvidia-gpulibs or nvptx-unknown-unknown
        # triple (before CUDA-10.x) and is compatible with all NVPTX variants.
        accepted_triples += ("nvptx64-nvidia-gpulibs", "nvptx-unknown-unknown")

    _validate_bitcode(
        path, llvm=llvm, accepted_triples=accepted_triples, bindings=bindings
    )
    return ResolvedLibrary(name, path, bindings)


def _find_llvm_libm_bitcode(llvm: LLVMToolchain, target: CompilationTarget) -> Path:
    candidates = [
        directory / target.target_triple / "libm.bc" for directory in llvm.library_dirs
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    searched = ", ".join(str(path) for path in candidates)
    raise LibraryUnavailableError(
        f"LLVM-libm bitcode was not found. Searched: {searched}. "
        "Provide an explicit bitcode_path for a custom installation"
    )


def _find_libdevice_bitcode(backend_sdk: BackendSDK) -> Path:
    if not isinstance(backend_sdk, CUDAToolkit):
        raise ValueError("libdevice requires a CUDA Toolkit")
    return backend_sdk.root / "nvvm" / "libdevice" / "libdevice.10.bc"


def _validate_bitcode(
    path: Path,
    *,
    llvm: LLVMToolchain,
    accepted_triples: tuple[str, ...],
    bindings: tuple[tuple[str, str], ...],
) -> None:
    if not path.is_file():
        raise LibraryUnavailableError(f"library bitcode was not found: {path}")

    ir = run_tool([llvm.llvm_dis, path, "-o", "-"])
    triple = re.search(r'^target triple = "([^"]+)"', ir, re.MULTILINE)
    if triple is None:
        raise ValueError(f"bitcode {path} does not declare a target triple")
    actual = triple.group(1)
    if actual not in accepted_triples:
        expected = " or ".join(repr(triple) for triple in accepted_triples)
        raise LibraryUnavailableError(
            f"bitcode {path} has target triple {actual!r}; expected {expected}"
        )

    if not bindings:
        return

    output = run_tool(
        [
            llvm.llvm_nm,
            "--defined-only",
            "--extern-only",
            "--just-symbol-name",
            path,
        ]
    )
    symbols = set(output.splitlines())
    missing = [
        f"{name} -> {symbol}" for name, symbol in bindings if symbol not in symbols
    ]
    if missing:
        raise LibraryUnavailableError(
            f"bitcode {path} does not define symbols for bindings: {', '.join(missing)}"
        )
