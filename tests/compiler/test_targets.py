# Copyright 2026 Leandro A. Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Tests for GPU compilation target selection."""

from pathlib import Path
from typing import cast

import pytest

from gpu_libm_bench.compiler import targets
from gpu_libm_bench.compiler.toolchains import CudaToolkit, LLVMToolchain
from gpu_libm_bench.devices import DeviceInfo, TimerInfo


@pytest.mark.parametrize("architecture", [None, "sm_90"])
def test_architecture_selection_uses_device_only_as_default(
    monkeypatch: pytest.MonkeyPatch, llvm: LLVMToolchain, architecture: str | None
) -> None:
    monkeypatch.setattr(targets, "run_tool", lambda command: "\tsm_90\n\tsm_120\n")
    target = targets.resolve_target(
        backend="cuda",
        target_triple="nvptx64-nvidia-cuda",
        architecture=architecture,
        llvm=llvm,
        device=DeviceInfo(
            "cuda",
            1,
            "GPU",
            "sm_120",
            1024,
            TimerInfo("%globaltimer", 64, 1_000_000_000),
        ),
    )
    assert target.architecture == (architecture or "sm_120")


@pytest.mark.parametrize(
    ("triple", "arch", "message"),
    [
        ("amdgcn-amd-amdhsa", "sm_120", "does not match backend"),
        ("nvptx64-nvidia-cuda", None, "selected GPU"),
        ("nvptx64-nvidia-cuda", "sm_999", "does not support"),
    ],
)
def test_invalid_compilation_target(
    monkeypatch: pytest.MonkeyPatch,
    llvm: LLVMToolchain,
    triple: str,
    arch: str | None,
    message: str,
) -> None:
    monkeypatch.setattr(targets, "run_tool", lambda command: "\tsm_120\n")
    with pytest.raises(ValueError, match=message):
        targets.resolve_target(
            backend="cuda", target_triple=triple, architecture=arch, llvm=llvm
        )


def test_assembler_must_also_support_selected_architecture(
    monkeypatch: pytest.MonkeyPatch, llvm: LLVMToolchain
) -> None:
    sdk = CudaToolkit(Path("/cuda"), Path("/cuda/bin/ptxas"), "12.0")
    monkeypatch.setattr(
        targets,
        "run_tool",
        lambda command: (
            "\tsm_120\n" if command[0] == llvm.clang else "Allowed: 'sm_90'"
        ),
    )
    with pytest.raises(ValueError, match="CUDA assembler"):
        targets.resolve_target(
            backend="cuda",
            target_triple="nvptx64-nvidia-cuda",
            architecture="sm_120",
            llvm=llvm,
            sdk=sdk,
        )


def test_unsupported_sdk_type_is_rejected(
    monkeypatch: pytest.MonkeyPatch, llvm: LLVMToolchain
) -> None:
    monkeypatch.setattr(targets, "run_tool", lambda command: "\tsm_120\n")
    with pytest.raises(TypeError, match="unsupported SDK type: object"):
        targets.resolve_target(
            backend="cuda",
            target_triple="nvptx64-nvidia-cuda",
            architecture="sm_120",
            llvm=llvm,
            sdk=cast(CudaToolkit, object()),
        )
