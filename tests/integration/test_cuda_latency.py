# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for dependent-call latency measurements."""

import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_module

from gpu_libm_bench import inputs
from gpu_libm_bench.benchmarks import LatencyBenchmark
from gpu_libm_bench.compiler.images import resolve_device_image
from gpu_libm_bench.compiler.targets import CompilationTarget
from gpu_libm_bench.configs import (
    ArgumentConfig,
    FunctionConfig,
    SignatureConfig,
    register_config_schema,
    validate_config,
)
from gpu_libm_bench.devices import (
    DeviceContext,
    DeviceImage,
    DeviceModule,
    LaunchConfig,
)
from gpu_libm_bench.kernels import KernelArgument, KernelKind, KernelRequest, KernelSpec
from gpu_libm_bench.types import Scalar

pytestmark = [pytest.mark.cuda, pytest.mark.execution]


@pytest.fixture
def context() -> Iterator[DeviceContext]:
    with DeviceContext() as value:
        yield value


@pytest.fixture(scope="module")
def function() -> FunctionConfig:
    register_config_schema()
    with initialize_config_module(config_module="gpu_libm_bench.configs.yaml"):
        config = compose(
            config_name="config",
            overrides=[
                "function=expf",
                "metric=latency",
                "run.libraries=[llvm-libm]",
                "backend=cuda",
            ],
        )

    return validate_config(config).function


@pytest.fixture
def module(context: DeviceContext) -> DeviceModule:
    info = context.device_info
    target = CompilationTarget("cuda", "nvptx64-nvidia-cuda", info.architecture)

    configured = os.environ.get("GPU_LIBM_BENCH_DEVICE_IMAGE")
    image = resolve_device_image(
        "llvm-libm",
        target=target,
        image_path=Path(configured) if configured is not None else None,
    )

    return context.load_module(image)


@pytest.fixture
def latency_module(context: DeviceContext) -> DeviceModule:
    info = context.device_info
    target = CompilationTarget("cuda", "nvptx64-nvidia-cuda", info.architecture)

    configured = os.environ.get("GPU_LIBM_BENCH_LATENCY_IMAGE")
    if configured is not None:
        path = Path(configured)
    else:
        image = resolve_device_image("llvm-libm", target=target)
        path = image.image_path.parent / "tests" / "latency.cubin"

    return context.load_module(DeviceImage(path, target, "tests", (), None))


@pytest.mark.parametrize("chain_length", [1, 512])
def test_latency_benchmark_preserves_evaluation_results(
    context: DeviceContext,
    module: DeviceModule,
    function: FunctionConfig,
    chain_length: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = np.array([-110, -80, -1, 0, 1, 80, 100], dtype=np.float32)
    expected = np.empty_like(source)

    reference_input = context.create_buffer(source.size, dtype=np.float32)
    reference_output = context.create_buffer(source.size, dtype=np.float32)

    context.copy_to_device(reference_input, source)

    evaluate = module.get_kernel(
        KernelRequest("llvm-libm", function.name, KernelKind.EVALUATE).symbol,
        spec=KernelSpec.from_config(function, kind=KernelKind.EVALUATE),
    )
    context.launch(
        evaluate,
        reference_input,
        reference_output,
        np.uint64(source.size),
        config=LaunchConfig(block=(32, 1, 1)),
    )
    context.copy_to_host(expected, reference_output)

    kernel = module.get_kernel(
        KernelRequest("llvm-libm", function.name, KernelKind.LATENCY).symbol,
        spec=KernelSpec.from_config(function, kind=KernelKind.LATENCY),
    )
    benchmark = LatencyBenchmark(
        context,
        kernel,
        chain_length=chain_length,
        repetitions=2,
        warmup_runs=3,
    )

    launch = context.launch
    observed = []

    def record_result(
        kernel,
        *arguments,
        config: LaunchConfig,
    ) -> None:
        launch(kernel, *arguments, config=config)

        value = np.empty(1, dtype=np.float32)
        context.copy_to_host(value, arguments[4])
        observed.append(value.view(np.uint32)[0])

    monkeypatch.setattr(context, "launch", record_result)
    benchmark.measure(inputs.InputBatch("central", {"x": source}))

    expected_results = np.concatenate((expected[:3], np.repeat(expected, 2)))
    np.testing.assert_array_equal(observed, expected_results.view(np.uint32))


@pytest.mark.parametrize("chain_length", [1, 7])
def test_all_calls_preserve_input_bits_across_passes(
    context: DeviceContext,
    latency_module: DeviceModule,
    function: FunctionConfig,
    chain_length: int,
) -> None:
    pass_count = 2

    parameters = KernelSpec.from_config(function, kind=KernelKind.LATENCY).arguments
    spec = KernelSpec(
        parameters[:-1]
        + (
            KernelArgument("observed_inputs", "uint64", True),
            KernelArgument("observed_count", "uint32", True),
        )
        + parameters[-1:]
    )
    kernel = latency_module.get_kernel("latencyProbe", spec=spec)

    x = context.create_buffer(1, dtype=np.float32)
    mask = context.create_buffer(1, dtype=np.uint32)
    elapsed = context.create_buffer(1, dtype=np.uint64)
    output = context.create_buffer(1, dtype=np.float32)

    count = chain_length * pass_count
    observed = context.create_buffer(count + 1, dtype=np.uint64)
    calls = context.create_buffer(1, dtype=np.uint32)

    trace = np.empty(count + 1, dtype=np.uint64)
    call_count = np.empty(1, dtype=np.uint32)
    result = np.empty(1, dtype=np.float32)

    context.copy_to_device(mask, np.zeros(1, dtype=np.uint32))

    # Include signed zeros, subnormals, infinities, and NaNs with payloads.
    for pattern in (
        0x00000000,
        0x80000000,
        0x00000001,
        0x80000001,
        0x7F800000,
        0xFF800000,
        0x7FC12345,
        0x7F812345,
    ):
        source = np.array([pattern], dtype=np.uint32).view(np.float32)
        context.copy_to_device(x, source)
        context.copy_to_device(
            observed, np.full(count + 1, 0xDEADBEEF, dtype=np.uint64)
        )

        context.launch(
            kernel,
            np.uint32(pass_count),
            np.uint32(chain_length),
            mask,
            elapsed,
            output,
            observed,
            calls,
            x,
            config=LaunchConfig(),
        )

        context.copy_to_host(trace, observed)
        context.copy_to_host(call_count, calls)
        context.copy_to_host(result, output)

        assert call_count[0] == count
        assert (trace[:count] == pattern).all()
        assert trace[count] == 0xDEADBEEF
        assert result.view(np.uint32)[0] == pattern


@pytest.mark.parametrize(
    ("symbol", "signature", "inputs"),
    [
        (
            "halfBinaryLatencyProbe",
            SignatureConfig(
                "_Float16",
                [
                    ArgumentConfig("x", "_Float16", "input"),
                    ArgumentConfig("y", "_Float16", "input"),
                ],
            ),
            (np.float16(-1.25), np.float16(-0.0)),
        ),
        (
            "doubleIntLatencyProbe",
            SignatureConfig(
                "double",
                [
                    ArgumentConfig("x", "double", "input"),
                    ArgumentConfig("exponent", "int", "input"),
                ],
            ),
            (np.float64(-1.25), np.int32(-7)),
        ),
    ],
)
def test_latency_preserves_all_arguments_for_scalar_signatures(
    context: DeviceContext,
    latency_module: DeviceModule,
    symbol: str,
    signature: SignatureConfig,
    inputs: tuple[Scalar, ...],
) -> None:
    pass_count, chain_length = 2, 7
    count = chain_length * pass_count
    sources = tuple(np.array([value]) for value in inputs)
    storage_dtype = np.dtype(f"uint{sources[0].dtype.itemsize * 8}")

    buffers = tuple(context.create_buffer(1, dtype=source.dtype) for source in sources)
    for buffer, source in zip(buffers, sources, strict=True):
        context.copy_to_device(buffer, source)

    mask = context.create_buffer(1, dtype=storage_dtype)
    elapsed = context.create_buffer(1, dtype=np.uint64)
    output = context.create_buffer(1, dtype=sources[0].dtype)
    trace = np.full(count * len(inputs) + 1, 0xDEADBEEF, dtype=np.uint64)
    observed = context.create_buffer(trace.size, dtype=trace.dtype)
    calls = context.create_buffer(1, dtype=np.uint32)

    context.copy_to_device(mask, np.zeros(1, dtype=storage_dtype))
    context.copy_to_device(observed, trace)

    parameters = KernelSpec.from_config(
        FunctionConfig(symbol, signature, domains={}), kind=KernelKind.LATENCY
    ).arguments
    spec = KernelSpec(
        parameters[: -len(inputs)]
        + (
            KernelArgument("observed_inputs", "uint64", True),
            KernelArgument("observed_count", "uint32", True),
        )
        + parameters[-len(inputs) :]
    )
    kernel = latency_module.get_kernel(symbol, spec=spec)

    context.launch(
        kernel,
        np.uint32(pass_count),
        np.uint32(chain_length),
        mask,
        elapsed,
        output,
        observed,
        calls,
        *buffers,
        config=LaunchConfig(),
    )

    call_count = np.empty(1, dtype=np.uint32)
    result = np.empty_like(sources[0])

    context.copy_to_host(trace, observed)
    context.copy_to_host(call_count, calls)
    context.copy_to_host(result, output)

    expected_inputs = np.array(
        [source.view(f"uint{source.dtype.itemsize * 8}")[0] for source in sources],
        dtype=np.uint64,
    )

    assert call_count[0] == count
    np.testing.assert_array_equal(trace[:-1], np.tile(expected_inputs, count))
    assert trace[-1] == 0xDEADBEEF
    np.testing.assert_array_equal(
        result.view(storage_dtype), sources[0].view(storage_dtype)
    )
