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

"""Tests for GPU compilation target selection."""

from pathlib import Path
from typing import cast

import pytest

from gpu_libm_bench.compiler import targets
from gpu_libm_bench.compiler.toolchains import CUDAToolkit, LLVMToolchain, ROCmCoreSDK
from gpu_libm_bench.devices import DeviceInfo, LaunchLimits, TimerInfo


@pytest.mark.parametrize(
    ("architecture", "device_architecture"),
    [(None, "sm_120"), ("sm_120", "sm_120"), ("sm_90", None)],
)
def test_architecture_selection(
    monkeypatch: pytest.MonkeyPatch,
    llvm: LLVMToolchain,
    architecture: str | None,
    device_architecture: str | None,
) -> None:
    monkeypatch.setattr(targets, "run_tool", lambda command: "\tsm_90\n\tsm_120\n")
    target = targets.resolve_target(
        backend="cuda",
        target_triple="nvptx64-nvidia-cuda",
        architecture=architecture,
        llvm=llvm,
        device_info=DeviceInfo(
            "cuda",
            1,
            "GPU",
            device_architecture,
            16,
            32,
            2_000_000_000,
            1024,
            0,
            LaunchLimits(
                (2**31 - 1, 65_535, 65_535), (1024, 1024, 64), 1024, 48 * 1024
            ),
            TimerInfo("%globaltimer", 64, 1_000_000_000),
        )
        if device_architecture is not None
        else None,
    )
    assert target.architecture == (architecture or device_architecture)


@pytest.mark.parametrize("backend_sdk", [None, ROCmCoreSDK(Path("/rocm"), "test")])
def test_hip_target_uses_generic_clang_resolution(
    monkeypatch: pytest.MonkeyPatch, llvm: LLVMToolchain, backend_sdk
) -> None:
    commands = []

    def supported_cpus(command):
        commands.append(command)
        return "\tgfx942\n"

    monkeypatch.setattr(targets, "run_tool", supported_cpus)

    target = targets.resolve_target(
        backend="hip",
        target_triple="amdgcn-amd-amdhsa",
        architecture="gfx942",
        llvm=llvm,
        backend_sdk=backend_sdk,
    )

    assert target == targets.CompilationTarget("hip", "amdgcn-amd-amdhsa", "gfx942")
    assert commands == [
        [llvm.clang, "--target=amdgcn-amd-amdhsa", "--print-supported-cpus"]
    ]


def test_target_architecture_must_match_selected_device(
    monkeypatch: pytest.MonkeyPatch, llvm: LLVMToolchain
) -> None:
    def unexpected_tool_call(command):
        pytest.fail("target mismatch must be rejected before querying tools")

    monkeypatch.setattr(targets, "run_tool", unexpected_tool_call)
    device_info = DeviceInfo(
        "cuda",
        0,
        "GPU",
        "sm_120",
        16,
        32,
        2_000_000_000,
        1024,
        0,
        LaunchLimits((2**31 - 1, 65_535, 65_535), (1024, 1024, 64), 1024, 48 * 1024),
        TimerInfo("%globaltimer", 64, 10**9),
    )

    with pytest.raises(
        ValueError, match="target architecture does not match the selected device"
    ):
        targets.resolve_target(
            backend="cuda",
            target_triple="nvptx64-nvidia-cuda",
            architecture="sm_90",
            llvm=llvm,
            device_info=device_info,
        )


@pytest.mark.parametrize(
    ("triple", "arch", "error", "message"),
    [
        ("amdgcn-amd-amdhsa", "sm_120", ValueError, "does not match backend"),
        ("nvptx64-nvidia-cuda", None, targets.TargetUnavailableError, "selected GPU"),
        (
            "nvptx64-nvidia-cuda",
            "sm_999",
            targets.TargetUnavailableError,
            "does not support",
        ),
    ],
)
def test_invalid_compilation_target(
    monkeypatch: pytest.MonkeyPatch,
    llvm: LLVMToolchain,
    triple: str,
    arch: str | None,
    error: type[ValueError],
    message: str,
) -> None:
    monkeypatch.setattr(targets, "run_tool", lambda command: "\tsm_120\n")
    with pytest.raises(error, match=message):
        targets.resolve_target(
            backend="cuda", target_triple=triple, architecture=arch, llvm=llvm
        )


def test_assembler_must_also_support_selected_architecture(
    monkeypatch: pytest.MonkeyPatch, llvm: LLVMToolchain
) -> None:
    backend_sdk = CUDAToolkit(Path("/cuda"), Path("/cuda/bin/ptxas"), "12.0")
    monkeypatch.setattr(
        targets,
        "run_tool",
        lambda command: (
            "\tsm_120\n" if command[0] == llvm.clang else "Allowed: 'sm_90'"
        ),
    )
    with pytest.raises(targets.TargetUnavailableError, match="CUDA assembler"):
        targets.resolve_target(
            backend="cuda",
            target_triple="nvptx64-nvidia-cuda",
            architecture="sm_120",
            llvm=llvm,
            backend_sdk=backend_sdk,
        )


def test_unsupported_sdk_type_is_rejected(
    monkeypatch: pytest.MonkeyPatch, llvm: LLVMToolchain
) -> None:
    monkeypatch.setattr(targets, "run_tool", lambda command: "\tsm_120\n")
    with pytest.raises(TypeError, match="unsupported backend SDK type: object"):
        targets.resolve_target(
            backend="cuda",
            target_triple="nvptx64-nvidia-cuda",
            architecture="sm_120",
            llvm=llvm,
            backend_sdk=cast(CUDAToolkit, object()),
        )


@pytest.mark.parametrize(
    ("backend", "backend_sdk", "error"),
    [
        ("cuda", ROCmCoreSDK(Path("/rocm"), "test"), ValueError),
        ("hip", CUDAToolkit(Path("/cuda"), Path("/cuda/ptxas"), "test"), ValueError),
    ],
)
def test_target_sdk_compatibility_precedes_tool_execution(
    monkeypatch: pytest.MonkeyPatch, llvm: LLVMToolchain, backend, backend_sdk, error
) -> None:
    def unexpected_tool_call(command):
        pytest.fail("incompatible backend and SDK must not execute tools")

    monkeypatch.setattr(targets, "run_tool", unexpected_tool_call)

    with pytest.raises(error):
        targets.resolve_target(
            backend=backend,
            target_triple=(
                "nvptx64-nvidia-cuda" if backend == "cuda" else "amdgcn-amd-amdhsa"
            ),
            architecture="sm_120" if backend == "cuda" else "gfx942",
            llvm=llvm,
            backend_sdk=backend_sdk,
        )
