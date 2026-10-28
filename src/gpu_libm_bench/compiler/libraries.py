# Copyright 2026 Leandro A. Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Device math library resolution and artifact validation."""

import re
from dataclasses import dataclass
from pathlib import Path

from ._commands import run_tool
from .targets import CompilationTarget
from .toolchains import LLVMToolchain


@dataclass(frozen=True, slots=True)
class DeviceLibrary:
    name: str
    bitcode_file: Path


def resolve_device_library(
    *,
    library: str,
    llvm: LLVMToolchain,
    target: CompilationTarget,
    bitcode_path: Path | None = None,
    required_symbols: tuple[str, ...] = (),
) -> DeviceLibrary:
    if library == "llvm-libm":
        return _resolve_llvm_libm(
            llvm=llvm,
            target=target,
            bitcode_path=bitcode_path,
            required_symbols=required_symbols,
        )
    else:
        raise ValueError(f"Unsupported device math library: {library!r}.")


def _resolve_llvm_libm(
    *,
    llvm: LLVMToolchain,
    target: CompilationTarget,
    bitcode_path: Path | None,
    required_symbols: tuple[str, ...],
) -> DeviceLibrary:
    if bitcode_path is None:
        bitcode_path = llvm.root / "lib" / target.target_triple / "libm.bc"
    path = bitcode_path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"LLVM-libm bitcode was not found: {path}")
    ir = run_tool([llvm.llvm_dis, path, "-o", "-"])
    triple = re.search(r'^target triple = "([^"]+)"', ir, re.MULTILINE)
    if triple is None or triple.group(1) != target.target_triple:
        actual = triple.group(1) if triple is not None else "missing"
        raise ValueError(
            f"Bitcode {path} has target triple {actual!r}; "
            f"expected {target.target_triple!r}."
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
                f"Bitcode {path} does not define symbols: {', '.join(sorted(missing))}."
            )
    return DeviceLibrary("llvm-libm", path)
