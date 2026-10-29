# Copyright 2026 Leandro A. Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Math library resolution and artifact validation."""

import re
from dataclasses import dataclass
from pathlib import Path

from ._commands import run_tool
from .targets import CompilationTarget
from .toolchains import LLVMToolchain


@dataclass(frozen=True, slots=True)
class Library:
    name: str
    bitcode_file: Path


def resolve_library(
    *,
    library: str,
    llvm: LLVMToolchain,
    target: CompilationTarget,
    bitcode_path: Path | None = None,
    required_symbols: tuple[str, ...] = (),
) -> Library:
    if library == "llvm-libm":
        return _resolve_llvm_libm(
            llvm=llvm,
            target=target,
            bitcode_path=bitcode_path,
            required_symbols=required_symbols,
        )
    else:
        raise ValueError(f"unsupported math library: {library!r}")


def _resolve_llvm_libm(
    *,
    llvm: LLVMToolchain,
    target: CompilationTarget,
    bitcode_path: Path | None,
    required_symbols: tuple[str, ...],
) -> Library:
    if bitcode_path is None:
        candidates = [
            directory / target.target_triple / "libm.bc"
            for directory in llvm.library_dirs
        ]
        bitcode_path = next((path for path in candidates if path.is_file()), None)
        if bitcode_path is None:
            searched = ", ".join(str(path) for path in candidates)
            raise FileNotFoundError(
                f"LLVM-libm bitcode was not found. Searched: {searched}. "
                "Provide an explicit bitcode_path for a custom installation"
            )

    path = bitcode_path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"LLVM-libm bitcode was not found: {path}")

    ir = run_tool([llvm.llvm_dis, path, "-o", "-"])
    triple = re.search(r'^target triple = "([^"]+)"', ir, re.MULTILINE)
    if triple is None or triple.group(1) != target.target_triple:
        actual = triple.group(1) if triple is not None else "missing"
        raise ValueError(
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
            raise ValueError(
                f"bitcode {path} does not define symbols: {', '.join(sorted(missing))}"
            )

    return Library("llvm-libm", path)
