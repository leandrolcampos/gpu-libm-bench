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

"""Compilation target resolution for GPU backends."""

import re
from dataclasses import dataclass

from gpu_libm_bench.backends import Backend
from gpu_libm_bench.devices import DeviceInfo

from ._commands import run_tool
from .toolchains import (
    BackendSDK,
    CUDAToolkit,
    LLVMToolchain,
    validate_backend_sdk_compatibility,
)

_TARGET_TRIPLES = {
    Backend.CUDA: "nvptx64-nvidia-cuda",
    Backend.HIP: "amdgcn-amd-amdhsa",
}


@dataclass(frozen=True, slots=True)
class CompilationTarget:
    backend: Backend
    target_triple: str
    architecture: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend", Backend(self.backend))


class TargetUnavailableError(ValueError):
    pass


def resolve_target(
    *,
    backend: Backend | str,
    target_triple: str,
    architecture: str | None,
    llvm: LLVMToolchain,
    device_info: DeviceInfo | None = None,
    backend_sdk: BackendSDK | None = None,
) -> CompilationTarget:
    backend = Backend(backend)

    if target_triple != _TARGET_TRIPLES[backend]:
        raise ValueError(
            f"target triple {target_triple!r} does not match backend {backend!r}"
        )
    if device_info is not None and device_info.backend != backend:
        raise ValueError(
            f"device backend {device_info.backend!r} does not match {backend!r}"
        )
    if architecture is None:
        if device_info is None:
            raise TargetUnavailableError(
                "architecture is unset; provide information for the selected GPU"
            )
        architecture = device_info.architecture

    if device_info is not None and architecture != device_info.architecture:
        raise ValueError("target architecture does not match the selected device")

    if backend_sdk is not None:
        validate_backend_sdk_compatibility(backend, backend_sdk)

    output = run_tool(
        [llvm.clang, f"--target={target_triple}", "--print-supported-cpus"]
    )
    supported = set(re.findall(r"^\s+((?:sm_|gfx)[\w-]+)\s*$", output, re.MULTILINE))
    if not supported:
        raise RuntimeError(
            f"Clang did not report GPU architectures for {target_triple}"
        )
    if architecture not in supported:
        raise TargetUnavailableError(
            f"Clang {llvm.clang} does not support GPU architecture {architecture!r}"
        )

    if isinstance(backend_sdk, CUDAToolkit):
        assembler_help = run_tool([backend_sdk.ptxas, "--help"])
        if architecture not in re.findall(r"'(sm_[\w]+)'", assembler_help):
            raise TargetUnavailableError(
                f"CUDA assembler {backend_sdk.ptxas} does not support {architecture!r}"
            )

    return CompilationTarget(backend, target_triple, architecture)
