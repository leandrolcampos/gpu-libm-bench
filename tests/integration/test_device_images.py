# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for math library calls, linking, and compiled device images."""

import json
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

import pytest

from gpu_libm_bench.compiler._commands import run_tool
from gpu_libm_bench.compiler.images import resolve_device_image
from gpu_libm_bench.compiler.targets import CompilationTarget
from gpu_libm_bench.compiler.toolchains import LLVMToolchain, resolve_llvm
from gpu_libm_bench.devices import DeviceImage
from gpu_libm_bench.kernels import KernelKind, KernelRequest

pytestmark = pytest.mark.images


@pytest.fixture(scope="module")
def compiled_image() -> tuple[DeviceImage, LLVMToolchain, dict[str, Any]]:
    configured = os.environ.get("GPU_LIBM_BENCH_DEVICE_IMAGE")
    if configured is not None:
        image_path = Path(configured).resolve()
    else:
        root = Path(__file__).resolve().parents[2] / "build" / "kernels"
        candidates = list(root.glob("cuda/*/llvm-libm/math.cubin"))
        if len(candidates) != 1:
            pytest.fail(
                "Build a device image or set GPU_LIBM_BENCH_DEVICE_IMAGE to select one."
            )
        image_path = candidates[0]

    metadata = json.loads(image_path.with_suffix(".manifest.json").read_text())
    target = CompilationTarget(
        metadata["backend"], metadata["target_triple"], metadata["architecture"]
    )
    image = resolve_device_image(
        metadata["library"], target=target, image_path=image_path
    )
    llvm = resolve_llvm(Path(metadata["llvm_root"]))

    return image, llvm, metadata


@pytest.fixture(scope="module")
def unlinked_command(
    compiled_image: tuple[DeviceImage, LLVMToolchain, dict[str, Any]],
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


def test_wrappers_preserve_math_function_calls(
    unlinked_command: list[str],
    tmp_path: Path,
) -> None:
    raw = tmp_path / "unlinked.ll"
    run_tool([*unlinked_command, "-S", "-emit-llvm", "-o", raw])

    raw_ir = raw.read_text()
    assert re.search(r"\bcall\b[^\n]*@expf\(", raw_ir)
    assert "@llvm.exp.f32(" not in raw_ir


def test_missing_library_cannot_be_replaced_by_a_builtin(
    compiled_image: tuple[DeviceImage, LLVMToolchain, dict[str, Any]],
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
    assert re.search(r"(?:undefined|unresolved)[^\n]*\bexpf\b", result.stderr, re.I)


def test_image_exports_registered_kernels(
    compiled_image: tuple[DeviceImage, LLVMToolchain, dict[str, Any]],
) -> None:
    image, llvm, metadata = compiled_image
    symbols = run_tool(
        [
            llvm.llvm_nm,
            "--defined-only",
            "--extern-only",
            "--just-symbol-name",
            image.image_path,
        ]
    ).splitlines()

    ptx = Path(metadata["ptx"]).read_text()
    assert re.search(
        rf"^\.target {re.escape(image.target.architecture)}(?:,|\s|$)",
        ptx,
        re.MULTILINE,
    )

    for function in metadata["functions"]:
        for kind in KernelKind:
            request = KernelRequest(image.library, function, kind)
            assert request.symbol in symbols
            assert re.search(
                rf"\.visible\s+\.entry\s+{re.escape(request.symbol)}\(", ptx
            )

    cuobjdump = Path(metadata["sdk_root"]) / "bin" / "cuobjdump"
    dump = run_tool([cuobjdump, "--dump-sass", image.image_path])
    assert f"code for {image.target.architecture}" in dump
