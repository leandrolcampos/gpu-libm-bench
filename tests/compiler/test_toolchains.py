# Copyright 2026 Leandro A. Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Tests for toolchain discovery precedence and coherence."""

import json
from pathlib import Path

import pytest

from gpu_libm_bench.compiler import toolchains


def _write_llvm_metadata(
    command: list[str | Path], llvm: toolchains.LLVMToolchain
) -> None:
    directory = Path(command[command.index("-B") + 1])
    (directory / "llvm-toolchain.json").write_text(
        json.dumps(
            {
                "clang": str(llvm.clang),
                "llvm_dis": str(llvm.llvm_dis),
                "llvm_nm": str(llvm.llvm_nm),
                "library_dirs": [str(path) for path in llvm.library_dirs],
            }
        )
    )


@pytest.mark.parametrize("source", ["explicit", "environment", "path"])
def test_llvm_selection_precedence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    llvm: toolchains.LLVMToolchain,
    source: str,
) -> None:
    other_bin = tmp_path / "other" / "bin"
    other_bin.mkdir(parents=True)
    for tool in (llvm.clang, llvm.llvm_dis, llvm.llvm_nm):
        alternative = other_bin / tool.name
        alternative.touch()
        alternative.chmod(0o755)
    monkeypatch.setenv(
        "LLVM_ROOT", str(llvm.root if source == "environment" else other_bin.parent)
    )
    if source == "path":
        monkeypatch.delenv("LLVM_ROOT")
        search = tmp_path / "search" / "bin"
        search.mkdir(parents=True)
        (search / "clang++").symlink_to(llvm.clang)
        discovered = search / "clang++"
    else:
        discovered = other_bin / "clang++"
    monkeypatch.setattr(
        toolchains.shutil,
        "which",
        lambda name: str(discovered) if name == "clang++" else "cmake",
    )
    commands = []

    def version(command: list[str | Path]) -> str:
        if command[0] == "cmake":
            assert f"-DLLVM_ROOT={llvm.root}" in command
            _write_llvm_metadata(command, llvm)
            return ""
        commands.append(command)
        return "LLVM version 24.0.0git"

    monkeypatch.setattr(toolchains, "run_tool", version)
    selected = toolchains.resolve_llvm(llvm.root if source == "explicit" else None)
    assert selected.root == llvm.root
    assert selected.library_dirs == llvm.library_dirs
    assert {command[0] for command in commands} == {
        llvm.clang,
        llvm.llvm_dis,
        llvm.llvm_nm,
    }


@pytest.mark.parametrize("source", ["explicit", "environment"])
def test_invalid_llvm_selection_does_not_fall_back(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    llvm: toolchains.LLVMToolchain,
    source: str,
) -> None:
    missing = tmp_path / "missing"
    monkeypatch.setenv(
        "LLVM_ROOT", str(missing if source == "environment" else llvm.root)
    )
    monkeypatch.setattr(toolchains.shutil, "which", lambda name: str(llvm.clang))
    with pytest.raises(FileNotFoundError, match=str(missing)):
        toolchains.resolve_llvm(missing if source == "explicit" else None)


def test_llvm_tools_must_have_coherent_versions(
    monkeypatch: pytest.MonkeyPatch, llvm: toolchains.LLVMToolchain
) -> None:
    def version(command: list[str | Path]) -> str:
        if command[0] == "cmake":
            _write_llvm_metadata(command, llvm)
            return ""
        return (
            "LLVM version 20.1.0"
            if command[0] == llvm.llvm_nm
            else "LLVM version 24.0.0"
        )

    monkeypatch.setattr(toolchains.shutil, "which", lambda name: "cmake")
    monkeypatch.setattr(toolchains, "run_tool", version)
    with pytest.raises(ValueError, match="incompatible versions"):
        toolchains.resolve_llvm(llvm.root)


@pytest.mark.parametrize("use_requested_root", [True, False])
def test_cuda_passes_explicit_selection_to_cmake_and_preserves_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, use_requested_root: bool
) -> None:
    requested = tmp_path / "cuda requested"
    requested.mkdir()
    selected = requested if use_requested_root else tmp_path / "other cuda"
    ptxas = selected / "bin" / "ptxas"
    ptxas.parent.mkdir(parents=True, exist_ok=True)
    ptxas.touch()
    ptxas.chmod(0o755)

    def probe(command: list[str | Path]) -> str:
        if command[0] == ptxas:
            return "ptxas: release 13.4"
        assert f"-DCUDAToolkit_ROOT={requested}" in command
        directory = Path(command[command.index("-B") + 1])
        (directory / "cuda-toolkit.json").write_text(
            json.dumps(
                {"root": str(selected), "ptxas": str(ptxas), "version": "13.4.92"}
            )
        )
        return ""

    monkeypatch.setattr(toolchains.shutil, "which", lambda name: "cmake")
    monkeypatch.setattr(toolchains, "run_tool", probe)
    if use_requested_root:
        toolkit = toolchains.resolve_cuda(requested)
        assert toolkit.root == requested
        assert toolkit.ptxas == ptxas
    else:
        with pytest.raises(ValueError, match="instead of requested"):
            toolchains.resolve_cuda(requested)
