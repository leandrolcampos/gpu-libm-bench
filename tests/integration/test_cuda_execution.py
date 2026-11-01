# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for CUDA memory transfers and kernel execution."""

import ctypes
import os
from collections.abc import Iterator
from dataclasses import replace
from math import prod
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_module

from gpu_libm_bench.compiler.images import resolve_device_image
from gpu_libm_bench.compiler.targets import CompilationTarget
from gpu_libm_bench.configs import register_config_schema, validate_config
from gpu_libm_bench.devices import (
    DeviceContext,
    DeviceImage,
    DeviceKernel,
    LaunchConfig,
)
from gpu_libm_bench.kernels import KernelKind, KernelRequest, KernelSpec

pytestmark = [pytest.mark.cuda, pytest.mark.execution]


@pytest.fixture(scope="module")
def image() -> DeviceImage:
    info = DeviceContext.get_device_info()
    target = CompilationTarget("cuda", "nvptx64-nvidia-cuda", info.architecture)
    configured = os.environ.get("GPU_LIBM_BENCH_DEVICE_IMAGE")

    return resolve_device_image(
        "llvm-libm",
        target=target,
        image_path=Path(configured) if configured is not None else None,
    )


@pytest.fixture(scope="module")
def spec() -> KernelSpec:
    register_config_schema()
    with initialize_config_module(config_module="gpu_libm_bench.configs.yaml"):
        config = compose(
            config_name="config",
            overrides=[
                "function=expf",
                "metric=latency",
                "library=llvm-libm",
                "backend=cuda",
            ],
        )

    return KernelSpec.from_config(
        validate_config(config).function, kind=KernelKind.EVALUATE
    )


@pytest.fixture
def context() -> Iterator[DeviceContext]:
    with DeviceContext() as value:
        yield value


def load_kernel(
    context: DeviceContext, image: DeviceImage, spec: KernelSpec
) -> DeviceKernel:
    module = context.load_module(image)
    request = KernelRequest("llvm-libm", "expf", KernelKind.EVALUATE)
    return module.get_kernel(request.symbol, spec=spec)


@pytest.mark.parametrize("dtype", [np.float32, np.uint64])
def test_copy_preserves_values_and_destination_tail(
    context: DeviceContext, dtype: type[np.float32] | type[np.uint64]
) -> None:
    source = np.arange(19, dtype=dtype)
    if dtype is np.uint64:
        source += np.uint64(1 << 63)
    else:
        source = source * np.float32(0.25) - np.float32(1)
    source.flags.writeable = False
    destination = np.full(source.size + 8, 99, dtype=dtype)
    initial = np.full(source.size + 5, 123, dtype=dtype)

    buffer = context.create_buffer(initial.size, dtype=source.dtype)
    context.copy_to_device(buffer, initial)

    context.copy_to_device(buffer, source)
    context.copy_to_host(destination, buffer)

    np.testing.assert_array_equal(destination[: source.size], source)
    np.testing.assert_array_equal(destination[source.size : initial.size], initial[-5:])
    assert (destination[initial.size :] == 99).all()


@pytest.mark.parametrize(
    "config",
    [
        LaunchConfig(grid=(2, 1, 1), block=(32, 1, 1)),
        LaunchConfig(grid=(2, 2, 1), block=(5, 3, 1)),
        LaunchConfig(grid=(2, 2, 2), block=(3, 2, 2)),
    ],
)
def test_evaluate_exp_preserves_tail(
    context: DeviceContext, image: DeviceImage, spec: KernelSpec, config: LaunchConfig
) -> None:
    count = prod(config.grid) * prod(config.block) - 7
    source = np.linspace(-2, 2, count, dtype=np.float32)
    destination = np.full(count + 11, np.nan, dtype=np.float32)

    x = context.create_buffer(source.size, dtype=source.dtype)
    y = context.create_buffer(destination.size, dtype=destination.dtype)
    context.copy_to_device(x, source)
    context.copy_to_device(y, destination)

    kernel = load_kernel(context, image, spec)
    context.launch(kernel, x, y, np.uint64(count), config=config)
    context.copy_to_host(destination, y)

    np.testing.assert_allclose(destination[:count], np.exp(source), rtol=4e-7)
    assert np.isnan(destination[count:]).all()


def test_launch_observes_the_latest_host_copy(
    context: DeviceContext, image: DeviceImage, spec: KernelSpec
) -> None:
    source = np.empty(1 << 16, dtype=np.float32)
    destination = np.empty_like(source)
    x = context.create_buffer(source.size, dtype=source.dtype)
    y = context.create_buffer(destination.size, dtype=destination.dtype)
    kernel = load_kernel(context, image, spec)
    config = LaunchConfig(grid=(source.size // 256, 1, 1), block=(256, 1, 1))

    for value in (-1, 1) * 16:
        source.fill(value)
        context.copy_to_device(x, source)
        context.launch(kernel, x, y, np.uint64(source.size), config=config)
        context.copy_to_host(destination, y)

        np.testing.assert_allclose(destination, np.exp(np.float32(value)), rtol=4e-7)


def test_copy_rejects_incompatible_arrays(context: DeviceContext) -> None:
    buffer = context.create_buffer(8, dtype=np.float32)

    with pytest.raises(ValueError, match="same dtype") as to_device_error:
        context.copy_to_device(buffer, np.zeros(8, dtype=np.float64))
    with pytest.raises(ValueError, match="same dtype") as to_host_error:
        context.copy_to_host(np.zeros(8, dtype=np.float64), buffer)
    assert str(to_device_error.value) == str(to_host_error.value)

    with pytest.raises(ValueError, match="capacity") as to_device_error:
        context.copy_to_device(buffer, np.zeros(9, dtype=np.float32))
    with pytest.raises(ValueError, match="capacity") as to_host_error:
        context.copy_to_host(np.zeros(7, dtype=np.float32), buffer)
    assert str(to_device_error.value) == str(to_host_error.value)

    with pytest.raises(ValueError, match="source.*contiguous"):
        context.copy_to_device(buffer, np.zeros(16, dtype=np.float32)[::2])
    with pytest.raises(ValueError, match="destination.*contiguous"):
        context.copy_to_host(np.zeros(16, dtype=np.float32)[::2], buffer)

    destination = np.zeros(8, dtype=np.float32)
    destination.flags.writeable = False
    with pytest.raises(ValueError, match="writable"):
        context.copy_to_host(destination, buffer)


def test_launch_rejects_incompatible_arguments(
    context: DeviceContext, image: DeviceImage, spec: KernelSpec
) -> None:
    kernel = load_kernel(context, image, spec)
    buffer = context.create_buffer(8, dtype=np.float32)
    config = LaunchConfig()

    with pytest.raises(ValueError, match="argument count"):
        context.launch(kernel, buffer, config=config)
    with pytest.raises(ValueError, match="wrong dtype"):
        context.launch(kernel, buffer, buffer, np.int32(8), config=config)
    with pytest.raises(TypeError, match="expects a device buffer"):
        context.launch(kernel, np.float32(0), buffer, np.uint64(8), config=config)


def test_module_reports_load_and_symbol_errors(
    context: DeviceContext, image: DeviceImage, spec: KernelSpec, tmp_path: Path
) -> None:
    with pytest.raises(FileNotFoundError, match="missing.cubin"):
        context.load_module(replace(image, path=tmp_path / "missing.cubin"))

    module = context.load_module(image)
    with pytest.raises(RuntimeError, match="missingKernel"):
        module.get_kernel("missingKernel", spec=spec)


def test_kernel_and_buffer_outlive_the_creating_session(
    image: DeviceImage, spec: KernelSpec
) -> None:
    source = np.array([-1, 0, 1], dtype=np.float32)
    destination = np.empty_like(source)

    with DeviceContext() as original:
        kernel = load_kernel(original, image, spec)
        buffer = original.create_buffer(source.size, dtype=source.dtype)
        original.copy_to_device(buffer, source)

    with pytest.raises(RuntimeError, match="closed"):
        original.create_buffer(1, dtype=np.float32)

    with DeviceContext() as context:
        output = context.create_buffer(destination.size, dtype=destination.dtype)
        context.launch(
            kernel,
            buffer,
            output,
            np.uint64(source.size),
            config=LaunchConfig(block=(32, 1, 1)),
        )
        context.copy_to_host(destination, output)

    np.testing.assert_allclose(destination, np.exp(source), rtol=4e-7)


def test_operations_restore_an_external_cuda_context(
    image: DeviceImage, spec: KernelSpec
) -> None:
    driver = ctypes.CDLL("libcuda.so.1")
    previous = ctypes.c_void_p()
    assert driver.cuCtxGetCurrent(ctypes.byref(previous)) == 0

    external = ctypes.c_void_p()
    assert driver.cuCtxCreate_v2(ctypes.byref(external), 0, 0) == 0

    try:
        source = np.zeros(1, dtype=np.float32)
        destination = np.empty_like(source)

        with DeviceContext() as context:
            buffer = context.create_buffer(1, dtype=source.dtype)
            kernel = load_kernel(context, image, spec)

            context.copy_to_device(buffer, source)
            context.launch(kernel, buffer, buffer, np.uint64(1), config=LaunchConfig())
            context.copy_to_host(destination, buffer)

            current = ctypes.c_void_p()
            assert driver.cuCtxGetCurrent(ctypes.byref(current)) == 0
            assert current.value == external.value

        assert driver.cuCtxGetCurrent(ctypes.byref(current)) == 0
        assert current.value == external.value
    finally:
        assert driver.cuCtxDestroy_v2(external) == 0
        assert driver.cuCtxSetCurrent(previous) == 0
