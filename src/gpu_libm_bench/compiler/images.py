# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Kernel compilation and device image resolution."""

import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gpu_libm_bench.devices import DeviceImage

from ._cmake import CMakeBuild, project_directory
from ._commands import run_tool
from .libraries import LibraryBitcode
from .targets import CompilationTarget
from .toolchains import CUDAToolkit, LLVMToolchain


@dataclass(frozen=True, slots=True)
class DeviceImageBuildInfo:
    compiler: Path
    compiler_version: str
    sdk_name: str
    sdk_version: str
    sdk_root: Path
    bitcode_path: Path
    image: DeviceImage
    compile_options: tuple[str, ...] = ()
    link_options: tuple[str, ...] = ()


def build_kernels(
    *,
    bitcode: LibraryBitcode,
    target: CompilationTarget,
    llvm: LLVMToolchain,
    sdk: CUDAToolkit,
    build_directory: Path | None = None,
    build_tests: bool = False,
) -> DeviceImage:
    build = configure_kernels(
        bitcode=bitcode,
        target=target,
        llvm=llvm,
        sdk=sdk,
        build_directory=build_directory,
        build_tests=build_tests,
    )

    build.build()

    return resolve_device_image(
        bitcode.library, target=target, build_directory=build.build_directory
    )


def configure_kernels(
    *,
    bitcode: LibraryBitcode,
    target: CompilationTarget,
    llvm: LLVMToolchain,
    sdk: CUDAToolkit,
    build_directory: Path | None = None,
    build_tests: bool = False,
) -> CMakeBuild:
    arguments = _image_cmake_arguments(
        bitcode=bitcode, target=target, llvm=llvm, sdk=sdk
    )

    project = project_directory()
    if build_directory is None:
        build_directory = _default_build_directory(bitcode.library, target=target)

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
        build_directory,
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


def resolve_device_image(
    library: str,
    *,
    target: CompilationTarget,
    image_path: Path | None = None,
    build_directory: Path | None = None,
) -> DeviceImage:
    explicit = image_path is not None
    if image_path is None:
        if build_directory is None:
            build_directory = _default_build_directory(library, target=target)

        if target.backend == "cuda":
            image_path = build_directory / "math.cubin"
        else:
            raise ValueError(f"unsupported device image backend: {target.backend!r}")

    image_path = image_path.expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"device image was not found: {image_path}")

    manifest_path = image_path.with_suffix(".manifest.json")
    manifest_exists = manifest_path.is_file()
    functions: tuple[str, ...] | None = None
    if manifest_exists:
        metadata = _read_image_manifest(manifest_path, library=library, target=target)

        if "functions" in metadata:
            functions = tuple(metadata["functions"])
    elif not explicit:
        raise FileNotFoundError(f"device image manifest was not found: {manifest_path}")

    return DeviceImage(
        image_path,
        target,
        library,
        functions,
        manifest_path if manifest_exists else None,
    )


def read_device_image_build_info(image: DeviceImage) -> DeviceImageBuildInfo:
    manifest_path = image.manifest_path
    if manifest_path is None:
        raise ValueError(f"device image has no build manifest: {image.image_path}")
    metadata = _read_image_manifest(
        manifest_path, library=image.library, target=image.target
    )

    for field in (
        "image_path",
        "compiler",
        "compiler_version",
        "sdk_root",
        "sdk_version",
        "bitcode_path",
    ):
        if not isinstance(metadata.get(field), str) or not metadata[field]:
            raise ValueError(
                f"device image manifest {manifest_path} "
                f"must declare {field} as a nonempty string"
            )

    for field in ("compile_flags", "link_flags"):
        if not isinstance(metadata.get(field), str):
            raise ValueError(
                f"device image manifest {manifest_path} "
                f"must declare {field} as a string"
            )

    for field in ("compile_options", "link_options"):
        values = metadata.get(field)
        if not isinstance(values, list) or any(
            not isinstance(value, str) for value in values
        ):
            raise ValueError(
                f"device image manifest {manifest_path} "
                f"must declare {field} as a list of strings"
            )

    if Path(metadata["image_path"]).resolve() != image.image_path.resolve():
        raise ValueError(
            f"device image manifest {manifest_path} "
            f"does not match image_path={str(image.image_path)!r}"
        )

    sdk_names = {"cuda": "CUDA Toolkit", "rocm": "ROCm"}
    if image.target.backend not in sdk_names:
        raise ValueError(f"unsupported SDK backend: {image.target.backend!r}")

    return DeviceImageBuildInfo(
        compiler=Path(metadata["compiler"]),
        compiler_version=metadata["compiler_version"],
        sdk_name=sdk_names[image.target.backend],
        sdk_version=metadata["sdk_version"],
        sdk_root=Path(metadata["sdk_root"]),
        bitcode_path=Path(metadata["bitcode_path"]),
        image=image,
        compile_options=_expand_options(
            metadata["compile_flags"], metadata["compile_options"]
        ),
        link_options=_expand_options(metadata["link_flags"], metadata["link_options"]),
    )


def read_registered_functions(
    build: CMakeBuild, *, library: str, target: CompilationTarget
) -> tuple[str, ...]:
    path = build.build_directory / "build-plan.json"
    metadata = json.loads(path.read_text())

    expected = {
        "library": library,
        "backend": target.backend,
        "target_triple": target.target_triple,
        "architecture": target.architecture,
    }
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise ValueError(
                f"kernel build plan {path} does not match {field}={value!r}"
            )

    functions = metadata.get("functions")
    if not isinstance(functions, list) or any(
        not isinstance(function, str) for function in functions
    ):
        raise ValueError(f"kernel build plan {path} must declare a list of functions")

    return tuple(functions)


def _image_cmake_arguments(
    *,
    bitcode: LibraryBitcode,
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
        "LIBRARY": bitcode.library,
        "BITCODE_PATH": str(bitcode.bitcode_path),
    }
    return tuple(f"-DGPU_LIBM_BENCH_{name}={value}" for name, value in values.items())


def _default_build_directory(library: str, *, target: CompilationTarget) -> Path:
    return (
        project_directory()
        / "build"
        / "kernels"
        / target.backend
        / target.architecture
        / library
    )


def _read_image_manifest(
    path: Path, *, library: str, target: CompilationTarget
) -> dict[str, Any]:
    metadata = json.loads(path.read_text())
    if not isinstance(metadata, dict):
        raise ValueError(f"device image manifest {path} must contain an object")

    expected = {
        "library": library,
        "backend": target.backend,
        "target_triple": target.target_triple,
        "architecture": target.architecture,
    }
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise ValueError(
                f"device image manifest {path} does not match {field}={value!r}"
            )

    return metadata


def _expand_options(flags: str, options: list[str]) -> tuple[str, ...]:
    arguments = shlex.split(flags)
    for option in options:
        if option.startswith("SHELL:"):
            arguments.extend(shlex.split(option.removeprefix("SHELL:")))
        else:
            arguments.append(option)

    return tuple(arguments)
