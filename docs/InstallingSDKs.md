# Building the LLVM Fork Used by Mage

This document outlines the steps to configure, build, and test the LLVM fork used by the Mage research project.

The process was validated on the following system:
* **OS:** Ubuntu 26.04.1 LTS
* **Kernel Version:** Linux 7.0.0-31-generic
* **Hardware Model**: ASUS ProArt X870E-CREATOR WIFI
* **Processor:** AMD Ryzen 9 9950X (32 cores)
* **Memory:** 32 GB
* **AMD GPU:** AMD Radeon RX 7700 XT (gfx1101)
* **NVIDIA GPU:** NVIDIA GeForce RTX 5070 (sm_120)

## 1. Prerequisites

Ensure you have already installed a recent NVIDIA driver on your host machine. This guide does not cover this step.

## 2. Install the CUDA Toolkit

Follow the official instructions provided on the [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads) page to install the toolkit for your system.

Verify that the NVIDIA driver is installed and that the GPU is visible:

```bash
nvidia-smi
```

## 3. Install the AMDGPU Stack

Follow the official instructions in the [AMD ROCm documentation](https://rocm.docs.amd.com/en/latest/install/rocm.html) to install the AMD GPU Driver (amdgpu) and the ROCm Full Suite.

Check if the AMD GPU Driver (amdgpu) is installed:

```bash
dkms status
```

Check if the GPU is listed as an available agent:

```bash
rocminfo
```

## 4. Set Up Environment Variables

Append the following paths to your `~/.bashrc` to ensure the build system can locate CUDA and the compiled LLVM toolchain.

```bash
echo 'export CUDA_HOME="/usr/local/cuda"' >> ~/.bashrc
echo 'export LLVM_ROOT="$HOME/opt/llvm"' >> ~/.bashrc
echo 'export PATH="$CUDA_HOME/bin:$LLVM_ROOT/bin${PATH:+:$PATH}"' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH="/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"' >> ~/.bashrc
```

Activate the changes and confirm the variable is set:

```bash
source ~/.bashrc
echo "$LLVM_ROOT"
```

## 5. Install the Basic Toolchain

Install the minimal host toolchain required to configure, build, and test the necessary LLVM subprojects.

```bash
sudo apt update
sudo apt -y install \
  build-essential \
  ccache \
  cmake \
  gcc-multilib \
  git \
  libmpfr-dev \
  ninja-build \
  python3 \
  python3-pip
```

## 6. Check Out the LLVM Fork

Clone the specific branch used by the Mage project. A shallow clone (`--depth=1`) is fast and sufficient for users who only need to build the toolchain:

```bash
git clone -b mage-staging --depth=1 https://github.com/leandrolcampos/llvm-project.git
```

## 7. Configure the LLVM Build

Below is a configuration recipe to generate a Release build of the LLVM subprojects _Clang_, _Offload_ and the _C library_. The C library is configured to target host CPUs, AMD GPUs, and NVIDIA GPUs.

```bash
cd llvm-project
cmake -S llvm -B build -G Ninja \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_DISABLE_PRECOMPILE_HEADERS=ON \
  -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra;lld" \
  -DLLVM_ENABLE_RUNTIMES="openmp;offload;libc" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_PARALLEL_LINK_JOBS=1 \
  -DCMAKE_INSTALL_PREFIX="$LLVM_ROOT" \
  -DRUNTIMES_amdgcn-amd-amdhsa_LLVM_ENABLE_RUNTIMES=libc \
  -DRUNTIMES_nvptx64-nvidia-cuda_LLVM_ENABLE_RUNTIMES=libc \
  -DLLVM_RUNTIME_TARGETS="default;amdgcn-amd-amdhsa;nvptx64-nvidia-cuda"
```

**Why these options?**

| CMake flag | Rationale |
| ---------- | --------- |
| `CMAKE_C_COMPILER_LAUNCHER=ccache` | Use ccache as a wrapper for the C compiler to speed up incremental builds by caching previous object files. |
| `CMAKE_CXX_COMPILER_LAUNCHER=ccache` | Use ccache as a wrapper for the C++ compiler to avoid redundant compilation of unchanged source files. |
| `CMAKE_DISABLE_PRECOMPILE_HEADERS=ON` | Disable precompiled headers to avoid friction with ccache. |
| `LLVM_ENABLE_PROJECTS="clang;clang-tools-extra;lld"` | Provide a recent Clang to compile the GPU C library, the `clangd` language server for IDE integration, and LLD to link GPU executables. |
| `LLVM_ENABLE_RUNTIMES="openmp;offload;libc"` | Include OpenMP (required by Offload), the Offload infrastructure itself, and the C library for the host. |
| `LLVM_ENABLE_ASSERTIONS=ON` | Keep assertion checks active even in a release build (see _Note_ below). |
| `LLVM_PARALLEL_LINK_JOBS=1` | Limit concurrent link jobs to prevent out-of-memory issues (see _Important_ below). |
| `RUNTIMES_amdgcn-amd-amdhsa_LLVM_ENABLE_RUNTIMES=libc` | Build the LLVM C library specifically for the AMD GPU target. |
| `RUNTIMES_nvptx64-nvidia-cuda_LLVM_ENABLE_RUNTIMES=libc` | Build the LLVM C library specifically for the NVIDIA GPU target. |
| `LLVM_RUNTIME_TARGETS="default;amdgcn-amd-amdhsa;nvptx64-nvidia-cuda"` | Set the enabled targets to build: the host CPU, the AMD GPU architecture, and the NVIDIA GPU architecture. |

> [!NOTE]
> `LLVM_ENABLE_ASSERTIONS=ON` keeps assertion checks active even in a release build (the default is `OFF`). Remove this flag if raw compile-time performance matters more than debugging safety.

> [!IMPORTANT]
> To avoid out-of-memory (OOM) issues during the link phase, configure `LLVM_PARALLEL_LINK_JOBS` to permit only one link job per 15 GB of available RAM on the host machine.

## 8. Build, Install, and Run Tests

After configuring the build system with the CMake command above, compile and install the required LLVM subprojects:

```bash
ninja -C build install -j 8
```

> [!NOTE]
> Running Ninja with high parallelism during the compilation phase can cause spurious failures, out-of-resource errors, or indefinite hangs. Limit the number of jobs with `-j <N>` if you encounter such issues.

Run the Offload API unit tests to ensure host-device communication is working:

```bash
ninja -C build/runtimes/runtimes-bins check-offload-unit -j 1
```

Finally, run all the tests for the GPU C library across both GPU architectures:

```bash
ninja -C build/runtimes/runtimes-amdgcn-amd-amdhsa-bins check-libc -j 1 -k 0
ninja -C build/runtimes/runtimes-nvptx64-nvidia-cuda-bins check-libc -j 1 -k 0
```
