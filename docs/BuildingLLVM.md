# Building LLVM for GPU Libm Bench

This guide builds Clang, LLD, development tools, and LLVM-libc from the official LLVM repository. LLVM-libc supplies the math library used by GPU Libm Bench.

The process was validated on the following system:

* **OS:** Ubuntu 26.04.1 LTS
* **Kernel Version:** Linux 7.0.0-42-generic
* **Hardware Model:** ASUS ProArt X870E-CREATOR WIFI
* **Processor:** AMD Ryzen 9 9950X (16 cores, 32 threads)
* **Memory:** 32 GB
* **AMD GPU:** AMD Radeon RX 7700 XT (gfx1101)
* **NVIDIA GPU:** NVIDIA GeForce RTX 5070 (sm_120)

## 1. Prerequisites

Install and configure the SDK for your GPU as described in [Installing GPU SDKs](InstallingSDKs.md). Only the SDK for the GPU vendor you intend to use is needed.

Install the host build tools and math test dependencies using your distribution's package manager. For example, on Ubuntu:

```bash
sudo apt update
sudo apt -y install \
  build-essential \
  ccache \
  cmake \
  git \
  libmpfr-dev \
  ninja-build \
  python3
```

## 2. Configure the Environment

Add these lines to `~/.bashrc` to set the installation directory and make the LLVM tools available after installation:

```bash
export LLVM_ROOT="$HOME/opt/llvm"
export PATH="$LLVM_ROOT/bin${PATH:+:$PATH}"
```

Load the changes in the current Bash session:

```bash
source ~/.bashrc
echo "$LLVM_ROOT"
```

GPU Libm Bench uses `LLVM_ROOT` to locate the installed toolchain.

## 3. Get the LLVM Sources

Clone the upstream default branch. A shallow clone is sufficient to build the toolchain:

```bash
git clone --depth=1 https://github.com/llvm/llvm-project.git
cd llvm-project
```

The default branch changes over time. Record the revision to identify the sources used for your build:

```bash
git rev-parse HEAD
```

## 4. Configure the Build

Build the compiler and libraries together, using the new Clang for the libraries. The `default` runtime target builds LLVM-libc for the host; the other two targets build it for AMD and NVIDIA GPUs. `clang-tools-extra` provides development tools such as `clangd` and `clang-tidy`.

The GPU libraries are cross-compiled, so both GPU targets can remain enabled even without GPU hardware or SDKs installed.

```bash
cmake -S llvm -B build -G Ninja \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_DISABLE_PRECOMPILE_HEADERS=ON \
  -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra;lld" \
  -DLLVM_ENABLE_RUNTIMES=libc \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_PARALLEL_LINK_JOBS=1 \
  -DCMAKE_INSTALL_PREFIX="$LLVM_ROOT" \
  -DRUNTIMES_amdgcn-amd-amdhsa_LLVM_ENABLE_RUNTIMES=libc \
  -DRUNTIMES_nvptx64-nvidia-cuda_LLVM_ENABLE_RUNTIMES=libc \
  -DLLVM_RUNTIME_TARGETS="default;amdgcn-amd-amdhsa;nvptx64-nvidia-cuda"
```

`ccache` speeds up rebuilds. Assertions remain enabled in the Release build to help diagnose failures, and one concurrent link job limits peak memory use.

For details on the GPU library build, see the [LLVM-libc build guide](https://libc.llvm.org/gpu/building.html).

## 5. Build and Install

```bash
ninja -C build install -j 8
```

Reduce `-j 8` if compilation exhausts host memory.

Verify the installed tools:

```bash
clang --version
llvm-dis --version
llvm-nm --version
clangd --version
clang-tidy --version
```

## 6. Run the GPU Library Tests

Tests require the corresponding GPU, driver/runtime, and LLVM GPU loader. Installing the SDK as recommended above provides the dependencies for the loader.

From the `llvm-project` directory, run only the tests for the GPU vendors available on your system.

For AMD GPUs with ROCm installed:

```bash
ninja -C build/runtimes/runtimes-amdgcn-amd-amdhsa-bins check-libc -j 1 -k 0
```

For NVIDIA GPUs with CUDA installed:

```bash
ninja -C build/runtimes/runtimes-nvptx64-nvidia-cuda-bins check-libc -j 1 -k 0
```

LLVM detects the local GPU architecture during configuration; check the CMake output to confirm that the GPU loaders and tests were enabled. See [Testing the GPU C library](https://libc.llvm.org/gpu/testing.html) for details.
