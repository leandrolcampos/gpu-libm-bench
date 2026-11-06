# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Tests for compilation configuration and command dispatch."""

import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from gpu_libm_bench.compiler import __main__ as build_cli
from gpu_libm_bench.compiler.libraries import Library
from gpu_libm_bench.compiler.targets import CompilationTarget
from gpu_libm_bench.compiler.toolchains import CUDAToolkit, LLVMToolchain
from gpu_libm_bench.devices import DeviceInfo, TimerInfo


def test_backend_command_requires_only_backend_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sdk = CUDAToolkit(tmp_path / "sdk", tmp_path / "sdk/bin/ptxas", "13.4")
    configure = Mock()

    monkeypatch.setattr(build_cli, "resolve_sdk", Mock(return_value=sdk))
    monkeypatch.setattr(build_cli, "resolve_llvm", Mock())
    monkeypatch.setattr(build_cli, "configure_backend", configure)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compiler",
            "backend",
            "--configure-only",
            "--build-tests",
            "--build-dir",
            str(tmp_path / "build"),
            "backend=cuda",
            f"backend.sdk_root={sdk.root}",
        ],
    )

    build_cli.main()

    build_cli.resolve_sdk.assert_called_once_with(backend="cuda", root=sdk.root)
    configure.assert_called_once_with(
        "cuda",
        sdk=sdk,
        build_directory=tmp_path / "build",
        build_tests=True,
    )
    build_cli.resolve_llvm.assert_not_called()


@pytest.mark.parametrize("architecture", [None, "sm_120"])
def test_kernel_command_preserves_build_selection(
    llvm: LLVMToolchain,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    architecture: str | None,
) -> None:
    sdk = CUDAToolkit(tmp_path / "sdk", tmp_path / "sdk/bin/ptxas", "13.4")
    device_info = DeviceInfo(
        "cuda",
        3,
        "test GPU",
        "sm_90",
        1024,
        TimerInfo("%globaltimer", 64, 1_000_000_000),
    )
    target = CompilationTarget(
        "cuda", "nvptx64-nvidia-cuda", architecture or device_info.architecture
    )
    library = Library("llvm-libm", tmp_path / "custom.bc")

    monkeypatch.setattr(build_cli, "resolve_sdk", Mock(return_value=sdk))
    monkeypatch.setattr(build_cli, "resolve_llvm", Mock(return_value=llvm))
    monkeypatch.setattr(build_cli, "resolve_target", Mock(return_value=target))
    monkeypatch.setattr(build_cli, "resolve_library", Mock(return_value=library))
    query = Mock(return_value=device_info)
    monkeypatch.setattr(build_cli.DeviceContext, "get_device_info", query)
    image_builder = Mock()
    monkeypatch.setattr(build_cli, "build_kernels", image_builder)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compiler",
            "kernels",
            "--build-dir",
            str(tmp_path / "build"),
            "--build-tests",
            "backend=cuda",
            f"backend.sdk_root={sdk.root}",
            "run.libraries=[llvm-libm]",
            "backend.device_id=3",
            f"backend.architecture={architecture or 'null'}",
            f"llvm.root={llvm.root}",
            f"libraries.llvm-libm.backends.cuda.bitcode_path={library.bitcode_file}",
        ],
    )

    build_cli.main()

    build_cli.resolve_sdk.assert_called_once_with(backend="cuda", root=sdk.root)
    build_cli.resolve_llvm.assert_called_once_with(llvm.root)

    if architecture is None:
        query.assert_called_once_with(
            device_info.device_id, backend=device_info.backend
        )
    else:
        query.assert_not_called()

    build_cli.resolve_target.assert_called_once_with(
        backend="cuda",
        target_triple=target.target_triple,
        architecture=architecture,
        llvm=llvm,
        device_info=device_info if architecture is None else None,
        sdk=sdk,
    )

    build_cli.resolve_library.assert_called_once_with(
        library=library.name,
        llvm=llvm,
        target=target,
        bitcode_path=library.bitcode_file,
    )

    image_builder.assert_called_once_with(
        library=library,
        target=target,
        llvm=llvm,
        sdk=sdk,
        build_directory=tmp_path / "build",
        build_tests=True,
    )


@pytest.mark.parametrize("selection", ["null", "[llvm-libm,other]"])
def test_kernel_command_requires_one_explicit_library(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, selection: str
) -> None:
    sdk = CUDAToolkit(tmp_path / "sdk", tmp_path / "sdk/bin/ptxas", "13.4")
    monkeypatch.setattr(build_cli, "resolve_sdk", Mock(return_value=sdk))
    monkeypatch.setattr(
        sys,
        "argv",
        ["compiler", "kernels", "backend=cuda", f"run.libraries={selection}"],
    )

    with pytest.raises(ValueError, match="exactly one name in run.libraries"):
        build_cli.main()
