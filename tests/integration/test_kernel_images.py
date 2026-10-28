# Copyright 2026 Leandro A. Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for compiled math kernel images."""

import json
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

import pytest

from gpu_libm_bench.compiler._commands import run_tool
from gpu_libm_bench.compiler.kernels import (
    KernelArtifact,
    resolve_kernel_artifact,
)
from gpu_libm_bench.compiler.targets import CompilationTarget
from gpu_libm_bench.compiler.toolchains import LLVMToolchain, resolve_llvm
from gpu_libm_bench.kernels import KernelKind, KernelRequest

pytestmark = pytest.mark.kernels


@pytest.fixture(scope="module")
def compiled_image() -> tuple[KernelArtifact, LLVMToolchain, dict[str, Any]]:
    configured = os.environ.get("GPU_LIBM_BENCH_KERNEL_IMAGE")
    if configured is not None:
        image = Path(configured).resolve()
    else:
        root = Path(__file__).resolve().parents[2] / "build" / "kernels"
        candidates = list(root.glob("cuda/*/llvm-libm/math.cubin"))
        if len(candidates) != 1:
            pytest.fail(
                "Build a device image or set GPU_LIBM_BENCH_KERNEL_IMAGE to select one."
            )
        image = candidates[0]
    metadata = json.loads(image.with_suffix(".manifest.json").read_text())
    target = CompilationTarget(
        metadata["backend"], metadata["target_triple"], metadata["architecture"]
    )
    request = KernelRequest(metadata["library"], "expf", KernelKind.EVALUATE)
    artifact = resolve_kernel_artifact(request, target=target, image_path=image)
    llvm = resolve_llvm(Path(metadata["compiler"]).parent.parent)
    return artifact, llvm, metadata


@pytest.fixture(scope="module")
def unlinked_command(
    compiled_image: tuple[KernelArtifact, LLVMToolchain, dict[str, Any]],
) -> list[str]:
    _, _, metadata = compiled_image
    database = json.loads(Path(metadata["compile_commands"]).read_text())
    entry = next(item for item in database if item["file"] == metadata["sources"][0])
    arguments = shlex.split(entry["command"])
    command = []
    index = 0
    while index < len(arguments):
        if arguments[index : index + 3] == [
            "-Xclang",
            "-mlink-builtin-bitcode",
            "-Xclang",
        ]:
            index += 4
        elif arguments[index] == "-o":
            index += 2
        elif arguments[index] == "-c":
            index += 1
        else:
            command.append(arguments[index])
            index += 1
    return command


def test_image_exports_registered_kernels(
    compiled_image: tuple[KernelArtifact, LLVMToolchain, dict[str, Any]],
) -> None:
    artifact, llvm, metadata = compiled_image
    symbols = run_tool(
        [
            llvm.llvm_nm,
            "--defined-only",
            "--extern-only",
            "--just-symbol-name",
            artifact.image,
        ]
    ).splitlines()
    ptx = Path(metadata["ptx"]).read_text()
    assert re.search(
        rf"^\.target {re.escape(artifact.target.architecture)}(?:,|\s|$)",
        ptx,
        re.MULTILINE,
    )
    for function in metadata["functions"]:
        request = KernelRequest(artifact.library, function, KernelKind.EVALUATE)
        assert request.symbol in symbols
        assert re.search(rf"\.visible\s+\.entry\s+{re.escape(request.symbol)}\(", ptx)
    cuobjdump = Path(metadata["sdk_root"]) / "bin" / "cuobjdump"
    dump = run_tool([cuobjdump, "--dump-sass", artifact.image])
    assert f"code for {artifact.target.architecture}" in dump


def test_selected_library_resolves_wrapper_calls(
    compiled_image: tuple[KernelArtifact, LLVMToolchain, dict[str, Any]],
    unlinked_command: list[str],
    tmp_path: Path,
) -> None:
    artifact, llvm, _ = compiled_image
    raw = tmp_path / "unlinked.ll"
    run_tool([*unlinked_command, "-S", "-emit-llvm", "-o", raw])
    raw_ir = raw.read_text()
    assert re.search(r"\bcall\b[^\n]*@expf\(", raw_ir)
    assert "@llvm.exp.f32(" not in raw_ir
    linked = list(artifact.image.parent.glob(f"{artifact.image.name}.*.precodegen.bc"))
    assert len(linked) == 1
    undefined = run_tool(
        [
            llvm.llvm_nm,
            "--undefined-only",
            "--just-symbol-name",
            linked[0],
        ]
    )
    assert not undefined


def test_missing_library_cannot_be_replaced_by_a_builtin(
    compiled_image: tuple[KernelArtifact, LLVMToolchain, dict[str, Any]],
    unlinked_command: list[str],
    tmp_path: Path,
) -> None:
    _, _, metadata = compiled_image
    command = [
        *unlinked_command,
        *metadata["link_options"],
        "-o",
        str(tmp_path / "unlinked.cubin"),
    ]
    result = subprocess.run(
        command, cwd=tmp_path, capture_output=True, text=True, check=False, timeout=30
    )
    assert result.returncode != 0
    assert "Undefined reference to 'expf'" in result.stderr
