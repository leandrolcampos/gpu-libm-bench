# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Resolve library bitcode and function bindings for device image builds."""

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
    bitcode_path: Path | None = None,
    required_symbols: tuple[str, ...] = (),
) -> ResolvedLibrary:
    name = library_config.name
    match name:
        case "llvm-libm":
            path = _resolve_llvm_libm_bitcode(
                llvm=llvm,
                target=target,
                bitcode_path=bitcode_path,
                required_symbols=required_symbols,
            )
        case _:
            raise ValueError(f"unsupported library: {name!r}")

    bindings = tuple(
        (function, library_config.symbol_for(function))
        for function in sorted(library_config.functions)
    )
    return ResolvedLibrary(name, path, bindings)


def _resolve_llvm_libm_bitcode(
    *,
    llvm: LLVMToolchain,
    target: CompilationTarget,
    bitcode_path: Path | None,
    required_symbols: tuple[str, ...],
) -> Path:
    if bitcode_path is None:
        candidates = [
            directory / target.target_triple / "libm.bc"
            for directory in llvm.library_dirs
        ]
        bitcode_path = next((path for path in candidates if path.is_file()), None)
        if bitcode_path is None:
            searched = ", ".join(str(path) for path in candidates)
            raise LibraryUnavailableError(
                f"LLVM-libm bitcode was not found. Searched: {searched}. "
                "Provide an explicit bitcode_path for a custom installation"
            )

    path = bitcode_path.expanduser().resolve()
    if not path.is_file():
        raise LibraryUnavailableError(f"LLVM-libm bitcode was not found: {path}")

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

    if required_symbols:
        output = run_tool(
            [
                llvm.llvm_nm,
                "--defined-only",
                "--extern-only",
                "--just-symbol-name",
                path,
            ]
        )

        missing = set(required_symbols) - set(output.splitlines())
        if missing:
            raise LibraryUnavailableError(
                f"bitcode {path} does not define symbols: {', '.join(sorted(missing))}"
            )

    return path
