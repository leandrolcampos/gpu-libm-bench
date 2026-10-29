# Copyright 2026 Leandro A. Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Kernel compilation and device image resolution."""

import json
from pathlib import Path

from gpu_libm_bench.devices import DeviceImage

from .libraries import Library
from .targets import CompilationTarget
from .toolchains import CudaToolkit, LLVMToolchain


def image_cmake_arguments(
    *,
    library: Library,
    target: CompilationTarget,
    llvm: LLVMToolchain,
    sdk: CudaToolkit,
) -> tuple[str, ...]:
    if target.backend != "cuda":
        raise ValueError("only CUDA device image builds are currently supported")
    values = {
        "LLVM_ROOT": str(llvm.root),
        "CLANG": str(llvm.clang),
        "SDK_ROOT": str(sdk.root),
        "SDK_VERSION": sdk.version,
        "TARGET_TRIPLE": target.target_triple,
        "ARCHITECTURE": target.architecture,
        "LIBRARY": library.name,
        "BITCODE_FILE": str(library.bitcode_file),
    }
    return tuple(f"-DGPU_LIBM_BENCH_{name}={value}" for name, value in values.items())


def resolve_device_image(
    library: str,
    *,
    target: CompilationTarget,
    image_path: Path | None = None,
    build_root: Path | None = None,
) -> DeviceImage:
    explicit = image_path is not None
    if image_path is None:
        if build_root is None:
            build_root = Path(__file__).resolve().parents[3] / "build" / "kernels"
        suffix = ".cubin" if target.backend == "cuda" else ".hsaco"
        image_path = (
            build_root
            / target.backend
            / target.architecture
            / library
            / f"math{suffix}"
        )
    image = image_path.expanduser().resolve()
    if not image.is_file():
        raise FileNotFoundError(f"device image was not found: {image}")
    manifest_path = image.with_suffix(".manifest.json")
    manifest: Path | None = None
    functions: tuple[str, ...] | None = None
    if manifest_path.is_file():
        manifest = manifest_path
        metadata = json.loads(manifest_path.read_text())
        expected = {
            "library": library,
            "backend": target.backend,
            "target_triple": target.target_triple,
            "architecture": target.architecture,
        }
        for field, value in expected.items():
            if metadata.get(field) != value:
                raise ValueError(
                    f"image manifest {manifest} does not match {field}={value!r}"
                )
        if "functions" in metadata:
            functions = tuple(metadata["functions"])
    elif not explicit:
        raise FileNotFoundError(f"device image manifest was not found: {manifest_path}")
    return DeviceImage(image, target, library, functions, manifest)
