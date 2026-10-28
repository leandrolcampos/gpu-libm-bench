# Copyright 2026 Leandro A. Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Device image build requests and artifact resolution."""

import json
from dataclasses import dataclass
from pathlib import Path

from gpu_libm_bench.kernels import KernelRequest

from .libraries import DeviceLibrary
from .targets import CompilationTarget
from .toolchains import CudaToolkit, LLVMToolchain


@dataclass(frozen=True, slots=True)
class KernelArtifact:
    image: Path
    target: CompilationTarget
    library: str
    manifest: Path | None


def kernel_cmake_arguments(
    *,
    library: DeviceLibrary,
    target: CompilationTarget,
    llvm: LLVMToolchain,
    sdk: CudaToolkit,
) -> tuple[str, ...]:
    if target.backend != "cuda":
        raise ValueError("Only CUDA kernel image builds are currently supported.")
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


def resolve_kernel_artifact(
    request: KernelRequest,
    *,
    target: CompilationTarget,
    image_path: Path | None = None,
    build_root: Path | None = None,
) -> KernelArtifact:
    explicit = image_path is not None
    if image_path is None:
        if build_root is None:
            build_root = Path(__file__).resolve().parents[3] / "build" / "kernels"
        suffix = ".cubin" if target.backend == "cuda" else ".hsaco"
        image_path = (
            build_root
            / target.backend
            / target.architecture
            / request.library
            / f"math{suffix}"
        )
    image = image_path.expanduser().resolve()
    if not image.is_file():
        raise FileNotFoundError(f"Device image was not found: {image}")
    manifest_path = image.with_suffix(".manifest.json")
    manifest: Path | None = None
    if manifest_path.is_file():
        manifest = manifest_path
        metadata = json.loads(manifest_path.read_text())
        expected = {
            "library": request.library,
            "backend": target.backend,
            "target_triple": target.target_triple,
            "architecture": target.architecture,
        }
        for field, value in expected.items():
            if metadata.get(field) != value:
                raise ValueError(
                    f"Image manifest {manifest} does not match {field}={value!r}."
                )
        if request.function not in metadata.get("functions", []):
            raise ValueError(
                f"Image manifest {manifest} does not contain function "
                f"{request.function!r}."
            )
    elif not explicit:
        raise FileNotFoundError(f"Device image manifest was not found: {manifest_path}")
    return KernelArtifact(image, target, request.library, manifest)
