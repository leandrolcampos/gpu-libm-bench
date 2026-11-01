# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Tests for device image selection and build requests."""

import json
from pathlib import Path

import pytest

from gpu_libm_bench.compiler.images import (
    image_cmake_arguments,
    resolve_device_image,
)
from gpu_libm_bench.compiler.libraries import Library
from gpu_libm_bench.compiler.targets import CompilationTarget
from gpu_libm_bench.compiler.toolchains import CudaToolkit, LLVMToolchain

TARGET = CompilationTarget("cuda", "nvptx64-nvidia-cuda", "sm_120")


@pytest.fixture
def image_path(tmp_path: Path) -> Path:
    image = tmp_path / "cuda" / "sm_120" / "llvm-libm" / "math.cubin"
    image.parent.mkdir(parents=True)
    image.touch()
    image.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "library": "llvm-libm",
                "backend": "cuda",
                "target_triple": TARGET.target_triple,
                "architecture": "sm_120",
                "functions": ["expf"],
            }
        )
    )
    return image


def test_build_request_preserves_selected_tools_and_libraries(
    llvm: LLVMToolchain, tmp_path: Path
) -> None:
    library = Library("llvm-libm", tmp_path / "libm.bc")
    sdk = CudaToolkit(Path("/custom cuda"), Path("/custom cuda/bin/ptxas"), "13.4")
    arguments = image_cmake_arguments(
        library=library, target=TARGET, llvm=llvm, sdk=sdk
    )
    assert f"-DGPU_LIBM_BENCH_LLVM_ROOT={llvm.root}" in arguments
    assert f"-DGPU_LIBM_BENCH_CLANG={llvm.clang}" in arguments
    assert "-DGPU_LIBM_BENCH_SDK_ROOT=/custom cuda" in arguments
    assert f"-DGPU_LIBM_BENCH_BITCODE_FILE={library.bitcode_file}" in arguments
    assert "-DGPU_LIBM_BENCH_ARCHITECTURE=sm_120" in arguments


@pytest.mark.parametrize("functions", [["expf", "logf"], []])
def test_default_image_preserves_declared_functions(
    tmp_path: Path, image_path: Path, functions: list[str]
) -> None:
    manifest = image_path.with_suffix(".manifest.json")
    metadata = json.loads(manifest.read_text())
    metadata["functions"] = functions
    manifest.write_text(json.dumps(metadata))
    image = resolve_device_image("llvm-libm", target=TARGET, build_root=tmp_path)
    assert image.path == image_path
    assert image.functions == tuple(functions)


def test_custom_image_selection_does_not_fall_back(
    tmp_path: Path, image_path: Path
) -> None:
    custom = image_path.with_name("custom.cubin")
    custom.touch()
    image = resolve_device_image(
        "llvm-libm", target=TARGET, image_path=custom, build_root=tmp_path
    )
    assert image.path == custom
    assert image.manifest is None
    assert image.functions is None
    custom.unlink()
    with pytest.raises(FileNotFoundError, match="custom.cubin"):
        resolve_device_image(
            "llvm-libm", target=TARGET, image_path=custom, build_root=tmp_path
        )


@pytest.mark.parametrize("field", ["architecture", "library"])
def test_image_provenance_must_match_selection(image_path: Path, field: str) -> None:
    manifest = image_path.with_suffix(".manifest.json")
    metadata = json.loads(manifest.read_text())
    metadata[field] = "other"
    manifest.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=f"does not match {field}"):
        resolve_device_image("llvm-libm", target=TARGET, image_path=image_path)
