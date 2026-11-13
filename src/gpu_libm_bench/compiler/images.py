# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Kernel compilation and device image resolution."""

import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gpu_libm_bench.devices import DeviceImage

from . import paths
from ._cmake import CMakeBuild
from ._commands import run_tool
from .libraries import ResolvedLibrary
from .targets import CompilationTarget
from .toolchains import CUDAToolkit, LLVMToolchain


@dataclass(frozen=True, slots=True)
class DeviceImageBuildInfo:
    compiler: Path
    compiler_version: str
    sdk_name: str
    sdk_version: str
    sdk_root: Path
    bitcode_path: Path | None
    image: DeviceImage
    compile_options: tuple[str, ...] = ()
    link_options: tuple[str, ...] = ()


def build_kernels(
    *,
    resolved_library: ResolvedLibrary,
    target: CompilationTarget,
    llvm: LLVMToolchain,
    sdk: CUDAToolkit,
    build_directory: Path | None = None,
    build_tests: bool = False,
) -> DeviceImage:
    build = configure_kernels(
        resolved_library=resolved_library,
        target=target,
        llvm=llvm,
        sdk=sdk,
        build_directory=build_directory,
        build_tests=build_tests,
    )

    build.build()

    return resolve_device_image(
        resolved_library.library, target=target, build_directory=build.build_directory
    )


def configure_kernels(
    *,
    resolved_library: ResolvedLibrary,
    target: CompilationTarget,
    llvm: LLVMToolchain,
    sdk: CUDAToolkit,
    build_directory: Path | None = None,
    build_tests: bool = False,
) -> CMakeBuild:
    arguments = _image_cmake_arguments(
        resolved_library=resolved_library, target=target, llvm=llvm, sdk=sdk
    )

    project = paths.source_root()
    if build_directory is None:
        build_directory = paths.image_build_directory(
            resolved_library.library,
            backend=target.backend,
            architecture=target.architecture,
        )

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
            build_directory = paths.image_build_directory(
                library, backend=target.backend, architecture=target.architecture
            )

        plan = build_directory / "build-plan.json"
        metadata = _read_image_metadata(
            plan, library=library, target=target, description="kernel build plan"
        )
        if (
            not isinstance(metadata.get("image_path"), str)
            or not metadata["image_path"]
        ):
            raise ValueError(
                f"kernel build plan {plan} must declare image_path as a nonempty string"
            )
        image_path = Path(metadata["image_path"])

    image_path = image_path.expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"device image was not found: {image_path}")

    manifest_path = image_path.with_suffix(".manifest.json")
    manifest_exists = manifest_path.is_file()
    functions: tuple[str, ...] | None = None
    if manifest_exists:
        metadata = _read_image_metadata(manifest_path, library=library, target=target)

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
    metadata = _read_image_metadata(
        manifest_path, library=image.library, target=image.target
    )

    for field in (
        "image_path",
        "compiler",
        "compiler_version",
        "sdk_root",
        "sdk_version",
    ):
        if not isinstance(metadata.get(field), str) or not metadata[field]:
            raise ValueError(
                f"device image manifest {manifest_path} "
                f"must declare {field} as a nonempty string"
            )

    bitcode_path = metadata.get("bitcode_path")
    if bitcode_path is not None and (
        not isinstance(bitcode_path, str) or not bitcode_path
    ):
        raise ValueError(
            f"device image manifest {manifest_path} "
            "must declare bitcode_path as a nonempty string or null"
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
        bitcode_path=Path(bitcode_path) if bitcode_path is not None else None,
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
    metadata = _read_image_metadata(
        path, library=library, target=target, description="kernel build plan"
    )

    functions = metadata.get("functions")
    if not isinstance(functions, list) or any(
        not isinstance(function, str) for function in functions
    ):
        raise ValueError(f"kernel build plan {path} must declare a list of functions")

    return tuple(functions)


def _image_cmake_arguments(
    *,
    resolved_library: ResolvedLibrary,
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
        "LIBRARY": resolved_library.library,
        "BITCODE_PATH": str(resolved_library.bitcode_path)
        if resolved_library.bitcode_path is not None
        else "",
    }
    return tuple(f"-DGPU_LIBM_BENCH_{name}={value}" for name, value in values.items())


def _read_image_metadata(
    path: Path,
    *,
    library: str,
    target: CompilationTarget,
    description: str = "device image manifest",
) -> dict[str, Any]:
    metadata = json.loads(path.read_text())
    if not isinstance(metadata, dict):
        raise ValueError(f"{description} {path} must contain an object")

    expected = {
        "library": library,
        "backend": target.backend,
        "target_triple": target.target_triple,
        "architecture": target.architecture,
    }
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise ValueError(f"{description} {path} does not match {field}={value!r}")

    return metadata


def _expand_options(flags: str, options: list[str]) -> tuple[str, ...]:
    arguments = shlex.split(flags)
    for option in options:
        if option.startswith("SHELL:"):
            arguments.extend(shlex.split(option.removeprefix("SHELL:")))
        else:
            arguments.append(option)

    return tuple(arguments)
