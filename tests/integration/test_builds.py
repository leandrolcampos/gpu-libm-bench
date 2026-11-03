# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for incremental native builds and artifact invalidation."""

import json
import os
import shlex
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from gpu_libm_bench.compiler import images
from gpu_libm_bench.compiler._cmake import CMakeBuild
from gpu_libm_bench.compiler._commands import run_tool
from gpu_libm_bench.compiler.backends import build_backend
from gpu_libm_bench.compiler.libraries import Library, resolve_library
from gpu_libm_bench.compiler.targets import resolve_target
from gpu_libm_bench.compiler.toolchains import resolve_cuda, resolve_llvm

pytestmark = pytest.mark.images


def test_backend_build_includes_native_tests(tmp_path: Path) -> None:
    sdk = resolve_cuda()
    build_backend("cuda", sdk=sdk, build_directory=tmp_path, build_tests=True)

    run_tool(
        [
            "ctest",
            "--test-dir",
            tmp_path,
            "--output-on-failure",
            "--no-tests=error",
        ]
    )


@pytest.fixture
def native_build(tmp_path: Path) -> CMakeBuild:
    source = tmp_path / "source"
    library = source / "lib"
    library.mkdir(parents=True)
    compiler = shutil.which("c++")
    assert compiler is not None
    wrapper = tmp_path / "c++"
    wrapper.write_text(f'#!/bin/sh\nexec {shlex.quote(compiler)} "$@"\n')
    wrapper.chmod(0o755)

    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.31)\n"
        "project(BuildValidation LANGUAGES CXX)\n"
        'set(OFFSET 0 CACHE STRING "Offset")\n'
        "add_subdirectory(lib)\n"
    )
    (library / "CMakeLists.txt").write_text(
        "add_library(example STATIC Example.cpp)\n"
        "target_compile_definitions(example PRIVATE OFFSET=${OFFSET})\n"
    )
    (library / "Example.cpp").write_text("int example() { return OFFSET; }\n")
    return CMakeBuild(
        source,
        tmp_path / "build",
        arguments=(f"-DCMAKE_CXX_COMPILER={wrapper}",),
    )


def test_build_reuses_artifacts_until_flags_change(
    native_build: CMakeBuild,
) -> None:
    native_build.configure()
    native_build.build()
    artifact = native_build.build_directory / "lib" / "libexample.a"
    timestamp = artifact.stat().st_mtime_ns

    native_build.configure()
    native_build.build()
    assert artifact.stat().st_mtime_ns == timestamp

    changed = replace(native_build, arguments=(*native_build.arguments, "-DOFFSET=2"))
    changed.configure()
    changed.build()
    assert artifact.stat().st_mtime_ns > timestamp


def test_compiler_change_triggers_rebuild_after_configure_only(
    native_build: CMakeBuild,
) -> None:
    native_build.configure()
    native_build.build()
    artifact = native_build.build_directory / "lib" / "libexample.a"
    timestamp = artifact.stat().st_mtime_ns

    wrapper = native_build.build_directory.parent / "c++"
    wrapper.write_text(wrapper.read_text() + "# compiler replacement\n")
    native_build.configure()
    repeated = replace(native_build)
    repeated.configure()

    repeated.build()
    assert artifact.stat().st_mtime_ns > timestamp


def test_device_image_rebuilds_when_bitcode_changes(tmp_path: Path) -> None:
    llvm = resolve_llvm()
    sdk = resolve_cuda()
    target = resolve_target(
        backend="cuda",
        target_triple="nvptx64-nvidia-cuda",
        architecture=os.environ.get("GPU_LIBM_BENCH_ARCHITECTURE", "sm_80"),
        llvm=llvm,
        sdk=sdk,
    )
    library = resolve_library(library="llvm-libm", llvm=llvm, target=target)

    first = tmp_path / "first.bc"
    second = tmp_path / "second.bc"
    shutil.copyfile(library.bitcode_file, first)
    shutil.copyfile(library.bitcode_file, second)

    build_directory = tmp_path / "build"

    def build(bitcode: Path):
        return images.build_kernels(
            library=Library(library.name, bitcode),
            target=target,
            llvm=llvm,
            sdk=sdk,
            build_directory=build_directory,
        )

    image = build(first)
    timestamp = image.path.stat().st_mtime_ns
    assert build(first).path.stat().st_mtime_ns == timestamp

    image = build(second)
    assert image.path.stat().st_mtime_ns > timestamp
    assert image.manifest is not None
    assert json.loads(image.manifest.read_text())["bitcode_file"] == str(second)
    timestamp = image.path.stat().st_mtime_ns

    os.utime(second, None)
    assert build(second).path.stat().st_mtime_ns > timestamp

    second.write_bytes(b"invalid bitcode")
    with pytest.raises(RuntimeError, match="command failed"):
        build(second)

    shutil.copyfile(first, second)
    build(second)
