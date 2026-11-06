# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Tests for device image selection and build requests."""

import json
from pathlib import Path
from typing import cast

import pytest

from gpu_libm_bench.compiler import images
from gpu_libm_bench.compiler._cmake import CMakeBuild
from gpu_libm_bench.compiler.images import (
    configure_kernels,
    read_registered_functions,
    resolve_device_image,
)
from gpu_libm_bench.compiler.libraries import Library
from gpu_libm_bench.compiler.targets import CompilationTarget
from gpu_libm_bench.compiler.toolchains import CUDAToolkit, LLVMToolchain

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


def test_kernel_configuration_preserves_tools_and_library(
    llvm: LLVMToolchain, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library = Library("llvm-libm", tmp_path / "libm.bc")
    sdk = CUDAToolkit(Path("/custom cuda"), Path("/custom cuda/bin/ptxas"), "13.4")

    monkeypatch.setattr(CMakeBuild, "configure", lambda self: None)
    monkeypatch.setattr(
        images, "run_tool", lambda command: "/llvm/clang-nvlink-wrapper"
    )

    build = configure_kernels(
        library=library,
        target=TARGET,
        llvm=llvm,
        sdk=sdk,
        build_directory=tmp_path / "build",
    )

    arguments = build.arguments
    assert f"-DGPU_LIBM_BENCH_LLVM_ROOT={llvm.root}" in arguments
    assert f"-DGPU_LIBM_BENCH_CLANG={llvm.clang}" in arguments
    assert "-DGPU_LIBM_BENCH_SDK_ROOT=/custom cuda" in arguments
    assert f"-DGPU_LIBM_BENCH_BITCODE_FILE={library.bitcode_file}" in arguments
    assert "-DGPU_LIBM_BENCH_ARCHITECTURE=sm_120" in arguments


@pytest.mark.parametrize(
    ("target", "sdk", "error", "message"),
    [
        (TARGET, object(), TypeError, "unsupported SDK type: object"),
        (
            CompilationTarget("rocm", "amdgcn-amd-amdhsa", "gfx1100"),
            CUDAToolkit(Path("/cuda"), Path("/cuda/bin/ptxas"), "13.4"),
            ValueError,
            "non-CUDA target",
        ),
    ],
)
def test_kernel_configuration_rejects_incompatible_sdk(
    llvm: LLVMToolchain,
    target: CompilationTarget,
    sdk: object,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        configure_kernels(
            library=Library("llvm-libm", Path("/libm.bc")),
            target=target,
            llvm=llvm,
            sdk=cast(CUDAToolkit, sdk),
        )


def test_default_image_preserves_declared_functions(
    tmp_path: Path, image_path: Path
) -> None:
    functions = ["expf", "logf"]
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


def test_kernel_support_comes_from_build_plan_before_compilation(
    tmp_path: Path, image_path: Path
) -> None:
    metadata = json.loads(image_path.with_suffix(".manifest.json").read_text())
    metadata["functions"] = ["logf"]
    image_path.with_name("build-plan.json").write_text(json.dumps(metadata))
    image_path.unlink()
    build = CMakeBuild(tmp_path / "sources", tmp_path)

    functions = read_registered_functions(build, library="llvm-libm", target=TARGET)

    assert functions == ("logf",)


@pytest.mark.parametrize("field", ["architecture", "functions"])
def test_invalid_kernel_build_plan_is_rejected(
    tmp_path: Path, image_path: Path, field: str
) -> None:
    metadata = json.loads(image_path.with_suffix(".manifest.json").read_text())
    metadata[field] = "invalid"
    image_path.with_name("build-plan.json").write_text(json.dumps(metadata))
    build = CMakeBuild(tmp_path / "sources", tmp_path)

    with pytest.raises(ValueError, match="kernel build plan"):
        read_registered_functions(build, library="llvm-libm", target=TARGET)
