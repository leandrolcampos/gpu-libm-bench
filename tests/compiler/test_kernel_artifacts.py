# Copyright 2026 Leandro A. Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Tests for device image selection and build requests."""

import json
from pathlib import Path

import pytest

from gpu_libm_bench.compiler.kernels import (
    kernel_cmake_arguments,
    resolve_kernel_artifact,
)
from gpu_libm_bench.compiler.libraries import DeviceLibrary
from gpu_libm_bench.compiler.targets import CompilationTarget
from gpu_libm_bench.compiler.toolchains import CudaToolkit, LLVMToolchain
from gpu_libm_bench.kernels import KernelKind, KernelRequest

TARGET = CompilationTarget("cuda", "nvptx64-nvidia-cuda", "sm_120")


@pytest.fixture
def request_and_image(tmp_path: Path) -> tuple[KernelRequest, Path]:
    request = KernelRequest("llvm-libm", "expf", KernelKind.EVALUATE)
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
    return request, image


def test_build_request_preserves_selected_tools_and_libraries(
    llvm: LLVMToolchain, tmp_path: Path
) -> None:
    library = DeviceLibrary("llvm-libm", tmp_path / "libm.bc")
    sdk = CudaToolkit(Path("/custom cuda"), Path("/custom cuda/bin/ptxas"), "13.4")
    arguments = kernel_cmake_arguments(
        library=library, target=TARGET, llvm=llvm, sdk=sdk
    )
    assert f"-DGPU_LIBM_BENCH_LLVM_ROOT={llvm.root}" in arguments
    assert "-DGPU_LIBM_BENCH_SDK_ROOT=/custom cuda" in arguments
    assert f"-DGPU_LIBM_BENCH_BITCODE_FILE={library.bitcode_file}" in arguments
    assert "-DGPU_LIBM_BENCH_ARCHITECTURE=sm_120" in arguments


def test_default_artifact_matches_request(
    tmp_path: Path, request_and_image: tuple[KernelRequest, Path]
) -> None:
    request, image = request_and_image
    artifact = resolve_kernel_artifact(request, target=TARGET, build_root=tmp_path)
    assert artifact.image == image


def test_custom_image_selection_does_not_fall_back(
    tmp_path: Path, request_and_image: tuple[KernelRequest, Path]
) -> None:
    request, _ = request_and_image
    custom = request_and_image[1].with_name("custom.cubin")
    custom.touch()
    artifact = resolve_kernel_artifact(
        request, target=TARGET, image_path=custom, build_root=tmp_path
    )
    assert artifact.image == custom
    assert artifact.manifest is None
    custom.unlink()
    with pytest.raises(FileNotFoundError, match="custom.cubin"):
        resolve_kernel_artifact(
            request, target=TARGET, image_path=custom, build_root=tmp_path
        )


@pytest.mark.parametrize("field", ["architecture", "library"])
def test_image_provenance_must_match_request(
    request_and_image: tuple[KernelRequest, Path], field: str
) -> None:
    request, image = request_and_image
    manifest = image.with_suffix(".manifest.json")
    metadata = json.loads(manifest.read_text())
    metadata[field] = "other"
    manifest.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=f"does not match {field}"):
        resolve_kernel_artifact(request, target=TARGET, image_path=image)


def test_image_must_contain_requested_function(
    request_and_image: tuple[KernelRequest, Path],
) -> None:
    request, image = request_and_image
    manifest = image.with_suffix(".manifest.json")
    metadata = json.loads(manifest.read_text())
    metadata["functions"] = ["logf"]
    manifest.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="does not contain function 'expf'"):
        resolve_kernel_artifact(request, target=TARGET, image_path=image)
