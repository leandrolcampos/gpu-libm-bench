# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Resolve math libraries and validate the bitcode required by their bindings."""

import re
from dataclasses import dataclass
from pathlib import Path

from gpu_libm_bench.configs import LibraryConfig

from ._commands import run_tool
from .targets import CompilationTarget
from .toolchains import LLVMToolchain


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
) -> ResolvedLibrary:
    bitcode_finders = {
        "llvm-libm": _find_llvm_libm_bitcode,
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
    path = find_bitcode(llvm, target) if bitcode_path is None else Path(bitcode_path)
    path = path.expanduser().resolve()

    bindings = tuple(
        (function, library_config.symbol_for(function))
        for function in sorted(library_config.functions)
    )
    _validate_bitcode(path, llvm=llvm, target=target, bindings=bindings)
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


def _validate_bitcode(
    path: Path,
    *,
    llvm: LLVMToolchain,
    target: CompilationTarget,
    bindings: tuple[tuple[str, str], ...],
) -> None:
    if not path.is_file():
        raise LibraryUnavailableError(f"library bitcode was not found: {path}")

    ir = run_tool([llvm.llvm_dis, path, "-o", "-"])
    triple = re.search(r'^target triple = "([^"]+)"', ir, re.MULTILINE)
    if triple is None:
        raise ValueError(f"bitcode {path} does not declare a target triple")
    if triple.group(1) != target.target_triple:
        actual = triple.group(1)
        raise LibraryUnavailableError(
            f"bitcode {path} has target triple {actual!r}; "
            f"expected {target.target_triple!r}"
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
