# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Test kernel bindings, image builds, image resolution, and build metadata."""

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from gpu_libm_bench.compiler import images
from gpu_libm_bench.compiler._cmake import CMakeBuild
from gpu_libm_bench.compiler.images import (
    configure_kernels,
    read_device_image_build_info,
    read_registered_functions,
    resolve_device_image,
)
from gpu_libm_bench.compiler.libraries import ResolvedLibrary
from gpu_libm_bench.compiler.targets import CompilationTarget
from gpu_libm_bench.compiler.toolchains import (
    BackendSDK,
    CUDAToolkit,
    LLVMToolchain,
    ROCmCoreSDK,
)
from gpu_libm_bench.configs import config as config_module

TARGET = CompilationTarget("cuda", "nvptx64-nvidia-cuda", "sm_120")


@pytest.fixture
def image_path(tmp_path: Path) -> Path:
    image_path = tmp_path / "math.cubin"
    image_path.touch()

    image_path.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "library": "llvm-libm",
                "backend": "cuda",
                "target_triple": TARGET.target_triple,
                "architecture": "sm_120",
                "functions": ["expf"],
                "image_path": str(image_path),
                "compiler": "/historical/llvm/clang++",
                "compiler_version": "clang version 19.0.1 (recorded)",
                "sdk_root": "/historical/cuda",
                "sdk_version": "12.4",
                "bitcode_path": "/historical/libm.bc",
                "compile_flags": '-flto -nogpulib -DNAME="two words"',
                "link_flags": "-nostdlib",
                "compile_options": [
                    "-O3",
                    "SHELL:-Xclang -mlink-builtin-bitcode -Xclang "
                    '"/historical/library with spaces.bc"',
                ],
                "link_options": ["-march=sm_120"],
            }
        )
    )
    image_path.with_name("build-plan.json").write_text(
        image_path.with_suffix(".manifest.json").read_text()
    )
    return image_path


def test_kernel_build_resolves_images_in_its_build_directory(
    llvm: LLVMToolchain,
    tmp_path: Path,
    image_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(CMakeBuild, "configure", lambda self: None)
    monkeypatch.setattr(CMakeBuild, "build", lambda self: None)
    monkeypatch.setattr(
        images, "run_tool", lambda command: "/llvm/clang-nvlink-wrapper"
    )

    image = images.build_kernels(
        resolved_library=ResolvedLibrary("llvm-libm", tmp_path / "libm.bc"),
        target=TARGET,
        llvm=llvm,
        backend_sdk=CUDAToolkit(Path("/cuda"), Path("/cuda/bin/ptxas"), "13.4"),
        build_directory=tmp_path,
    )

    assert image.image_path == image_path


def test_default_kernel_build_is_isolated_and_keeps_images_discoverable(
    llvm: LLVMToolchain, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(images.paths, "build_root", lambda: tmp_path / "build")
    monkeypatch.setattr(CMakeBuild, "configure", lambda self: None)
    monkeypatch.setattr(
        images, "run_tool", lambda command: "/llvm/clang-nvlink-wrapper"
    )
    backend_sdk = CUDAToolkit(Path("/cuda"), Path("/cuda/bin/ptxas"), "13.4")

    build = configure_kernels(
        resolved_library=ResolvedLibrary("llvm-libm", tmp_path / "libm.bc"),
        target=TARGET,
        llvm=llvm,
        backend_sdk=backend_sdk,
    )

    root = tmp_path / "build" / "kernels"
    assert build.source_directory == images.paths.source_root() / "kernels"
    assert build.build_directory == root / "cuda" / "sm_120" / "llvm-libm"

    plan = build.build_directory / "build-plan.json"
    plan.parent.mkdir(parents=True)
    plan.write_text(
        json.dumps(
            {
                "library": "llvm-libm",
                "backend": "cuda",
                "architecture": "sm_120",
                "target_triple": TARGET.target_triple,
                "functions": ["expf"],
                "image_path": str(plan.with_name("math.cubin")),
            }
        )
    )
    assert read_registered_functions(build, library="llvm-libm", target=TARGET) == (
        "expf",
    )

    image_path = plan.with_name("math.cubin")
    image_path.touch()
    image_path.with_suffix(".manifest.json").write_text(plan.read_text())
    image = resolve_device_image("llvm-libm", target=TARGET)
    assert image.image_path == image_path


def test_configure_kernels_refreshes_bindings_when_inputs_change(
    llvm: LLVMToolchain, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library = ResolvedLibrary("llvm-libm", tmp_path / "libm.bc", (("expf", "my_expf"),))
    function_path = tmp_path / "function" / "expf.yaml"
    function_path.parent.mkdir()
    function = (
        "signature:\n"
        "  return_type: float\n"
        "  arguments: [{name: x, type: float, role: input}]\n"
    )
    function_path.write_text(function)
    monkeypatch.setattr(config_module, "files", lambda package: tmp_path)
    monkeypatch.setattr(CMakeBuild, "configure", lambda self: None)
    monkeypatch.setattr(
        images, "run_tool", lambda command: "/llvm/clang-nvlink-wrapper"
    )

    def configure(resolved_library: ResolvedLibrary) -> Path:
        build = configure_kernels(
            resolved_library=resolved_library,
            target=TARGET,
            llvm=llvm,
            backend_sdk=CUDAToolkit(Path("/cuda"), Path("/cuda/bin/ptxas"), "13.4"),
            build_directory=tmp_path / "build",
        )
        return build.build_directory / "generated" / "LibraryBindings.def"

    bindings = configure(library)
    original = bindings.read_text()
    assert original == (
        "// Generated from library bindings and function signatures. Do not edit.\n\n"
        "#if defined(LIBM_BIND_EXPF)\n"
        "#define LIBM_NAME expf\n"
        "#define LIBM_SYMBOL my_expf\n"
        "#define LIBM_RETURN_TYPE float\n"
        "#define LIBM_ARG_TYPE float\n"
        "#endif\n"
    )

    os.utime(bindings, ns=(0, 0))
    timestamp = bindings.stat().st_mtime_ns
    function_path.write_text(function + "domains: {}\n")
    assert configure(library).stat().st_mtime_ns == timestamp

    function_path.write_text(function.replace("float", "double"))
    assert configure(library).read_text() == original.replace(" float\n", " double\n")

    assert not configure(replace(library, bindings=())).exists()


def test_configure_kernels_updates_editor_database(
    llvm: LLVMToolchain, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(images.paths, "build_root", lambda: tmp_path / "build")
    monkeypatch.setattr(
        images, "run_tool", lambda command: "/llvm/clang-nvlink-wrapper"
    )
    root = images.paths.images_build_root()
    aggregate = root / "compile_commands.json"
    source = str(tmp_path / "Profiling.cpp")

    def write_database(directory: Path, architecture: str) -> dict[str, str]:
        directory.mkdir(parents=True, exist_ok=True)
        entry = {
            "directory": str(directory),
            "file": source,
            "command": f"clang++ -march={architecture} -c {source}",
        }
        (directory / "compile_commands.json").write_text(json.dumps([entry]))
        return entry

    def configure(build: CMakeBuild) -> None:
        write_database(build.build_directory, TARGET.architecture)

    monkeypatch.setattr(CMakeBuild, "configure", configure)

    def configure_support(directory: Path | None = None) -> CMakeBuild:
        return configure_kernels(
            resolved_library=ResolvedLibrary("support"),
            target=TARGET,
            llvm=llvm,
            backend_sdk=CUDAToolkit(Path("/cuda"), Path("/cuda/bin/ptxas"), "13.4"),
            build_directory=directory,
        )

    build = configure_support()
    original = json.loads((build.build_directory / aggregate.name).read_text())
    assert json.loads(aggregate.read_text()) == original

    other = write_database(root / "hip" / "gfx1100" / "support", "gfx1100")
    configure_support(build.build_directory)
    assert json.loads(aggregate.read_text()) == [*original, other]

    timestamp = aggregate.stat().st_mtime_ns
    configure_support(tmp_path / "isolated")
    assert (tmp_path / "isolated" / aggregate.name).is_file()
    assert aggregate.stat().st_mtime_ns == timestamp


@pytest.mark.parametrize("library", ["llvm-libm", "support"])
def test_kernel_configuration_preserves_tools_and_library(
    llvm: LLVMToolchain,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    library: str,
) -> None:
    bitcode_path = tmp_path / "libm.bc" if library == "llvm-libm" else None
    resolved_library = ResolvedLibrary(library, bitcode_path)
    backend_sdk = CUDAToolkit(
        Path("/custom cuda"), Path("/custom cuda/bin/ptxas"), "13.4"
    )

    monkeypatch.setattr(CMakeBuild, "configure", lambda self: None)
    monkeypatch.setattr(
        images, "run_tool", lambda command: "/llvm/clang-nvlink-wrapper"
    )

    build = configure_kernels(
        resolved_library=resolved_library,
        target=TARGET,
        llvm=llvm,
        backend_sdk=backend_sdk,
        build_directory=tmp_path / "build",
    )

    arguments = build.arguments
    assert f"-DGPU_LIBM_BENCH_LLVM_ROOT={llvm.root}" in arguments
    assert f"-DGPU_LIBM_BENCH_CLANG={llvm.clang}" in arguments
    assert "-DGPU_LIBM_BENCH_SDK_ROOT=/custom cuda" in arguments
    assert f"-DGPU_LIBM_BENCH_LIBRARY={library}" in arguments
    assert f"-DGPU_LIBM_BENCH_BITCODE_PATH={bitcode_path or ''}" in arguments
    assert "-DGPU_LIBM_BENCH_ARCHITECTURE=sm_120" in arguments


@pytest.mark.parametrize(
    ("target", "backend_sdk", "error", "message"),
    [
        (TARGET, object(), TypeError, "unsupported backend SDK type: object"),
        (
            CompilationTarget("hip", "amdgcn-amd-amdhsa", "gfx1100"),
            CUDAToolkit(Path("/cuda"), Path("/cuda/bin/ptxas"), "13.4"),
            ValueError,
            "non-cuda backend",
        ),
        (
            TARGET,
            ROCmCoreSDK(Path("/rocm"), "test"),
            ValueError,
            "non-hip backend",
        ),
        (
            CompilationTarget("hip", "amdgcn-amd-amdhsa", "gfx1100"),
            ROCmCoreSDK(Path("/rocm"), "test"),
            NotImplementedError,
            "hip device image compilation is not implemented yet",
        ),
    ],
)
def test_kernel_configuration_rejects_invalid_or_unimplemented_combinations(
    monkeypatch: pytest.MonkeyPatch,
    llvm: LLVMToolchain,
    target: CompilationTarget,
    backend_sdk: object,
    error: type[Exception],
    message: str,
) -> None:
    def unexpected_configuration(self):
        pytest.fail("invalid or unimplemented combinations must not configure CMake")

    monkeypatch.setattr(CMakeBuild, "configure", unexpected_configuration)

    with pytest.raises(error, match=message):
        configure_kernels(
            resolved_library=ResolvedLibrary("llvm-libm", Path("/libm.bc")),
            target=target,
            llvm=llvm,
            backend_sdk=cast(BackendSDK, backend_sdk),
        )


def test_default_image_preserves_declared_functions(
    tmp_path: Path, image_path: Path
) -> None:
    functions = ["expf", "logf"]
    manifest_path = image_path.with_suffix(".manifest.json")
    metadata = json.loads(manifest_path.read_text())
    metadata["functions"] = functions
    manifest_path.write_text(json.dumps(metadata))

    image = resolve_device_image("llvm-libm", target=TARGET, build_directory=tmp_path)

    assert image.image_path == image_path
    assert image.functions == tuple(functions)


def test_custom_image_selection_does_not_fall_back(
    tmp_path: Path, image_path: Path
) -> None:
    custom = image_path.with_name("custom.cubin")
    custom.touch()

    image = resolve_device_image(
        "llvm-libm", target=TARGET, image_path=custom, build_directory=tmp_path
    )

    assert image.image_path == custom
    assert image.manifest_path is None

    custom.unlink()
    with pytest.raises(FileNotFoundError, match="custom.cubin"):
        resolve_device_image(
            "llvm-libm", target=TARGET, image_path=custom, build_directory=tmp_path
        )


@pytest.mark.parametrize("field", ["architecture", "library"])
def test_image_provenance_must_match_selection(image_path: Path, field: str) -> None:
    manifest_path = image_path.with_suffix(".manifest.json")
    metadata = json.loads(manifest_path.read_text())
    metadata[field] = "other"
    manifest_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match=f"does not match {field}"):
        resolve_device_image("llvm-libm", target=TARGET, image_path=image_path)


def test_build_info_uses_recorded_tools_and_all_flags(
    image_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = resolve_device_image("llvm-libm", target=TARGET, image_path=image_path)

    def unexpected_command(*args, **kwargs):
        pytest.fail("reading build provenance must not invoke tools")

    monkeypatch.setattr(images, "run_tool", unexpected_command)
    image_build_info = read_device_image_build_info(image)

    assert image_build_info.image == image
    assert image_build_info.compiler == Path("/historical/llvm/clang++")
    assert image_build_info.compiler_version == "clang version 19.0.1 (recorded)"

    assert (
        image_build_info.sdk_name,
        image_build_info.sdk_version,
        image_build_info.sdk_root,
    ) == (
        "CUDA Toolkit",
        "12.4",
        Path("/historical/cuda"),
    )
    assert image_build_info.bitcode_path == Path("/historical/libm.bc")

    assert image_build_info.compile_options == (
        "-flto",
        "-nogpulib",
        "-DNAME=two words",
        "-O3",
        "-Xclang",
        "-mlink-builtin-bitcode",
        "-Xclang",
        "/historical/library with spaces.bc",
    )
    assert image_build_info.link_options == ("-nostdlib", "-march=sm_120")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("compiler_version", None),
        ("compile_options", "-O3"),
        ("image_path", "/other.cubin"),
        ("architecture", "sm_90"),
        ("bitcode_path", ""),
    ],
)
def test_build_info_rejects_invalid_or_incompatible_metadata(
    image_path: Path, field: str, value: object
) -> None:
    image = resolve_device_image("llvm-libm", target=TARGET, image_path=image_path)
    assert image.manifest_path is not None

    metadata = json.loads(image.manifest_path.read_text())
    metadata[field] = value
    image.manifest_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match=field):
        read_device_image_build_info(image)


def test_build_info_requires_a_manifest(image_path: Path) -> None:
    image = resolve_device_image("llvm-libm", target=TARGET, image_path=image_path)
    with pytest.raises(ValueError, match="no build manifest"):
        read_device_image_build_info(replace(image, manifest_path=None))


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
