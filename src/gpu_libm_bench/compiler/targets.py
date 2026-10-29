# Copyright 2026 Leandro A. Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Compilation target resolution for GPU backends."""

import re
from dataclasses import dataclass

from gpu_libm_bench.devices import DeviceInfo

from ._commands import run_tool
from .toolchains import CudaToolkit, LLVMToolchain

_TARGET_TRIPLES = {
    "cuda": "nvptx64-nvidia-cuda",
    "rocm": "amdgcn-amd-amdhsa",
}


@dataclass(frozen=True, slots=True)
class CompilationTarget:
    backend: str
    target_triple: str
    architecture: str


def resolve_target(
    *,
    backend: str,
    target_triple: str,
    architecture: str | None,
    llvm: LLVMToolchain,
    device: DeviceInfo | None = None,
    sdk: CudaToolkit | None = None,
) -> CompilationTarget:
    if backend not in _TARGET_TRIPLES:
        raise ValueError(f"unsupported compilation backend: {backend!r}")
    if target_triple != _TARGET_TRIPLES[backend]:
        raise ValueError(
            f"target triple {target_triple!r} does not match backend {backend!r}"
        )
    if device is not None and device.backend != backend:
        raise ValueError(
            f"device backend {device.backend!r} does not match {backend!r}"
        )
    if architecture is None:
        if device is None:
            raise ValueError(
                "architecture is unset; provide information for the selected GPU"
            )
        architecture = device.architecture

    output = run_tool(
        [llvm.clang, f"--target={target_triple}", "--print-supported-cpus"]
    )
    supported = set(re.findall(r"^\s+((?:sm_|gfx)[\w-]+)\s*$", output, re.MULTILINE))
    if not supported:
        raise RuntimeError(
            f"Clang did not report GPU architectures for {target_triple}"
        )
    if architecture not in supported:
        raise ValueError(
            f"Clang {llvm.clang} does not support GPU architecture {architecture!r}"
        )

    if sdk is not None:
        if isinstance(sdk, CudaToolkit):
            if backend != "cuda":
                raise ValueError("a CUDA Toolkit cannot be used for a non-CUDA target")
            assembler_help = run_tool([sdk.ptxas, "--help"])
            if architecture not in re.findall(r"'(sm_[\w]+)'", assembler_help):
                raise ValueError(
                    f"CUDA assembler {sdk.ptxas} does not support {architecture!r}"
                )
        else:
            raise TypeError(f"unsupported SDK type: {type(sdk).__name__}")

    return CompilationTarget(backend, target_triple, architecture)
