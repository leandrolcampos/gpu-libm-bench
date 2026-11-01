# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Configuration of GPU kernel builds."""

import argparse
import os
from pathlib import Path

from gpu_libm_bench.compiler._commands import run_tool
from gpu_libm_bench.compiler.images import image_cmake_arguments
from gpu_libm_bench.compiler.libraries import resolve_library
from gpu_libm_bench.compiler.targets import resolve_target
from gpu_libm_bench.compiler.toolchains import resolve_cuda, resolve_llvm
from gpu_libm_bench.devices import DeviceContext


def main() -> None:
    parser = argparse.ArgumentParser(description="Configure a GPU math kernel build.")
    parser.add_argument("--llvm-root", type=Path)
    parser.add_argument("--sdk-root", type=Path)
    parser.add_argument(
        "--architecture", default=os.environ.get("GPU_LIBM_BENCH_ARCHITECTURE")
    )
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--bitcode-path", type=Path)
    parser.add_argument("--build-dir", type=Path, default=Path("build/kernels"))
    parser.add_argument("--build-testing", action="store_true")
    arguments = parser.parse_args()
    llvm = resolve_llvm(arguments.llvm_root)
    sdk = resolve_cuda(arguments.sdk_root)
    device = (
        DeviceContext.get_device_info(arguments.device_id)
        if arguments.architecture is None
        else None
    )
    target = resolve_target(
        backend="cuda",
        target_triple="nvptx64-nvidia-cuda",
        architecture=arguments.architecture,
        llvm=llvm,
        device=device,
        sdk=sdk,
    )
    library = resolve_library(
        library="llvm-libm",
        llvm=llvm,
        target=target,
        bitcode_path=arguments.bitcode_path,
    )
    source = Path(__file__).resolve().parents[1] / "kernels"
    output = run_tool(
        [
            "cmake",
            "-S",
            source,
            "-B",
            arguments.build_dir,
            "-G",
            "Ninja",
            *image_cmake_arguments(library=library, target=target, llvm=llvm, sdk=sdk),
            f"-DBUILD_TESTING={'ON' if arguments.build_testing else 'OFF'}",
        ]
    )
    print(output)


if __name__ == "__main__":
    main()
