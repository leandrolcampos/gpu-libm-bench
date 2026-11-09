# GPU Libm Bench

GPU Libm Bench is a benchmarking framework for evaluating GPU math libraries across functions, input domains, and hardware architectures. It aims to support reproducible comparisons, help researchers and developers understand trade-offs between numerical accuracy and performance, and guide optimization.

> [!WARNING]
> This project is being developed as part of PhD research and is still at an early stage. Its interfaces, supported configurations, and benchmarking methodologies are subject to change.

The current implementation measures the latency of `expf` from LLVM-libc's math library (`llvm-libm`) on NVIDIA GPUs using the CUDA backend.

## Environment Setup

The project currently targets Linux x86-64 and uses [Pixi](https://pixi.sh/) to manage its Python dependencies and native build tools.

1. Install and configure the SDK for your GPU following [Installing GPU SDKs](docs/InstallingSDKs.md). The current benchmark backend requires an NVIDIA GPU and CUDA.
2. Build and install the LLVM toolchain and LLVM-libc's math library for host and GPUs following [Building LLVM](docs/BuildingLLVM.md). Keep `LLVM_ROOT` set to the installation directory.
3. Install Pixi, clone this repository, and prepare the `native` environment:

```bash
git clone https://github.com/leandrolcampos/gpu-libm-bench.git
cd gpu-libm-bench
pixi install -e native
```

## Usage

Run benchmarks from the repository root, selecting the function, metric, and backend:

```bash
pixi run -e native gpu-libm-bench --details function=expf metric=latency backend=cuda
```

The command builds the required native backend and device images as needed, then runs the benchmark. `--details` adds experiment configuration and provenance to the report.

For example, a run on an NVIDIA GeForce RTX 5070 produced the following latency results (summary, in ns/call):

| Domain | Input distribution | Library | Mean | StDev | Min | Max | Samples |
| ------ | ------------------ | ------- | ---- | ----- | --- | --- | ------- |
| central | uniform(-10, 10) | llvm-libm | 50.011 | 0.1092 | 49.941 | 50.195 | 8192 |
| wide | uniform(-80, 80) | llvm-libm | 50.192 | 0.0013257 | 50.188 | 50.195 | 8192 |

## Contributing

This is currently a research project, and external contributions are not being accepted at this stage.

## License

This repository is licensed under the [Apache License 2.0](LICENSE).
