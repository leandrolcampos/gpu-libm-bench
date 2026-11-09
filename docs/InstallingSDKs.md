# Installing GPU SDKs

This guide covers GPU SDK installation and the corresponding environment configuration for GPU Libm Bench on Linux. Each section can be followed independently for the corresponding GPU vendor.

The environment setup below assumes Bash.

The process was validated on the following system:

* **OS:** Ubuntu 26.04.1 LTS
* **Kernel Version:** Linux 7.0.0-42-generic
* **Hardware Model:** ASUS ProArt X870E-CREATOR WIFI
* **Processor:** AMD Ryzen 9 9950X (16 cores, 32 threads)
* **Memory:** 32 GB
* **AMD GPU:** AMD Radeon RX 7700 XT (gfx1101)
* **NVIDIA GPU:** NVIDIA GeForce RTX 5070 (sm_120)

## 1. Install the CUDA Toolkit

### Install the SDK

A compatible NVIDIA driver is required and may already be installed by your Linux distribution. Check with `nvidia-smi`. If the driver is missing or incompatible with your chosen CUDA version, follow the [NVIDIA Driver Installation Guide](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/) to install or update it, then reboot.

Install the CUDA Toolkit using the [CUDA Downloads](https://developer.nvidia.com/cuda-downloads) selector and the [CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/). Select your operating system and architecture.

### Configure and Load the Environment

Add the following lines to `~/.bashrc`, adjusting `CUDA_HOME` to match your installation:

```bash
export CUDA_HOME="/usr/local/cuda"
export PATH="$CUDA_HOME/bin${PATH:+:$PATH}"
```

If you installed the CUDA Toolkit using the standalone `.run` installer, also add its library directory:

```bash
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Load the changes in the current Bash session:

```bash
source ~/.bashrc
```

### Verify the Installation

Run these commands as your regular user:

```bash
nvidia-smi
nvcc --version
```

`nvidia-smi` should list the NVIDIA GPU, and `nvcc` should report the installed toolkit version. The CUDA version shown by `nvidia-smi` indicates driver compatibility, not the installed toolkit version.

## 2. Install the ROCm Core SDK

### Install the SDK

A compatible AMD GPU driver (`amdgpu`) is required and may already be installed by your Linux distribution. Follow the [AMD ROCm installation guide](https://rocm.docs.amd.com/en/latest/install/rocm.html) to determine whether it needs to be installed or updated; reboot if you install or update it.

Install the complete ROCm Core SDK using the same guide for your GPU and operating system, selecting the **ROCm Full Suite** use case. Complete its post-installation steps, including GPU access permissions.

### Configure and Load the Environment

Add the following lines to `~/.bashrc`, adjusting `ROCM_PATH` to match your installation:

```bash
export ROCM_PATH="/opt/rocm"
export PATH="$ROCM_PATH/bin${PATH:+:$PATH}"
export LD_LIBRARY_PATH="$ROCM_PATH/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Load the changes in the current Bash session:

```bash
source ~/.bashrc
```

### Verify the Installation

Run these commands as your regular user:

```bash
rocminfo
hipcc --version
```

`rocminfo` should list the AMD GPU as a GPU agent, confirming that the driver and runtime can access it. `hipcc` should report the installed HIP compiler version.
