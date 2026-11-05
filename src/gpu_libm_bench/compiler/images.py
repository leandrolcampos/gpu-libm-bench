# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Kernel compilation and device image resolution."""

import json
from dataclasses import dataclass
from pathlib import Path

from gpu_libm_bench.devices import DeviceImage

from ._cmake import CMakeBuild, project_directory
from ._commands import run_tool
from .libraries import Library
from .targets import CompilationTarget
from .toolchains import CUDAToolkit, LLVMToolchain


@dataclass(frozen=True, slots=True)
class ImageBuildInfo:
    compiler: Path
    compiler_version: str
    sdk_name: str
    sdk_version: str
    sdk_root: Path
    bitcode_file: Path
    image: DeviceImage
    compile_options: tuple[str, ...] = ()
    link_options: tuple[str, ...] = ()


def configure_kernels(
    *,
    library: Library,
    target: CompilationTarget,
    llvm: LLVMToolchain,
    sdk: CUDAToolkit,
    build_directory: Path | None = None,
    build_tests: bool = False,
) -> CMakeBuild:
    arguments = _image_cmake_arguments(
        library=library, target=target, llvm=llvm, sdk=sdk
    )
    project = project_directory()
    toolchain_files: tuple[Path, ...] = (
        llvm.clang,
        project / "kernels" / "cmake" / "modules" / "GPULibmBenchToolchain.cmake",
    )
    if isinstance(sdk, CUDAToolkit):
        toolchain_files += (
            sdk.ptxas,
            sdk.ptxas.parent / "nvlink",
            Path(run_tool([llvm.clang, "--print-prog-name=clang-nvlink-wrapper"])),
        )

    build = CMakeBuild(
        project / "kernels",
        build_directory
        if build_directory is not None
        else project / "build" / "kernels",
        arguments=(*arguments, f"-DBUILD_TESTING={'ON' if build_tests else 'OFF'}"),
        toolchain=(
            str(llvm.root),
            llvm.version,
            str(sdk.root),
            sdk.version,
            target.target_triple,
        ),
        toolchain_files=toolchain_files,
    )

    build.configure()
    return build


def build_kernels(
    *,
    library: Library,
    target: CompilationTarget,
    llvm: LLVMToolchain,
    sdk: CUDAToolkit,
    build_directory: Path | None = None,
    build_tests: bool = False,
) -> DeviceImage:
    build = configure_kernels(
        library=library,
        target=target,
        llvm=llvm,
        sdk=sdk,
        build_directory=build_directory,
        build_tests=build_tests,
    )
    build.build()

    return resolve_device_image(
        library.name, target=target, build_root=build.build_directory
    )


def _image_cmake_arguments(
    *,
    library: Library,
    target: CompilationTarget,
    llvm: LLVMToolchain,
    sdk: CUDAToolkit,
) -> tuple[str, ...]:
    if isinstance(sdk, CUDAToolkit):
        if target.backend != "cuda":
            raise ValueError("a CUDA Toolkit cannot be used for a non-CUDA target")
    else:
        raise TypeError(f"unsupported SDK type: {type(sdk).__name__}")

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
            build_root = project_directory() / "build" / "kernels"
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
