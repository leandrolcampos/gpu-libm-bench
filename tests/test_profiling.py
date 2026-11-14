# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Tests for kernel profiling argument conversion and validation."""

from enum import Enum
from unittest.mock import Mock

import numpy as np
import pytest

from gpu_libm_bench.devices import (
    DeviceBuffer,
    DeviceContext,
    KernelMetric,
    KernelProfile,
    LaunchConfig,
)
from gpu_libm_bench.devices import context as context_module


@pytest.fixture
def context(monkeypatch: pytest.MonkeyPatch) -> DeviceContext:
    backend = Mock()
    backend.KernelMetric = Enum(
        "NativeKernelMetric", {"CLOCK_FREQUENCY": 0, "DURATION": 1}
    )
    backend.DeviceContext.return_value.released = False
    monkeypatch.setattr(
        "gpu_libm_bench.devices.context._load_backend", lambda name: backend
    )
    return DeviceContext()


@pytest.mark.parametrize(
    "metrics",
    [
        {KernelMetric.CLOCK_FREQUENCY},
        {KernelMetric.DURATION},
        {KernelMetric.CLOCK_FREQUENCY, KernelMetric.DURATION},
    ],
)
def test_profile_converts_arguments_metrics_and_result(
    context: DeviceContext, metrics: set[KernelMetric]
) -> None:
    backend = context_module._load_backend(context.backend)
    kernel = Mock()
    buffer = DeviceBuffer(Mock(), np.dtype(np.float32))
    scalar = np.uint64(1)
    config = LaunchConfig(block=(32, 1, 1))
    result = {
        "clock_frequency_hz": 2e9 if KernelMetric.CLOCK_FREQUENCY in metrics else None,
        "duration_ms": 0.25 if KernelMetric.DURATION in metrics else None,
    }
    context._native.profile.return_value = result

    profile = context.profile(kernel, buffer, scalar, config=config, metrics=metrics)

    assert profile == KernelProfile(**result)
    context._native.profile.assert_called_once_with(
        kernel._native,
        (buffer._native, scalar),
        config.grid,
        config.block,
        config.shared_memory_bytes,
        {backend.KernelMetric[metric.name] for metric in metrics},
    )
    kernel._validate_arguments.assert_called_once_with((buffer, scalar))
    context._native.launch.assert_not_called()


def test_profiling_rejects_empty_metrics(context: DeviceContext) -> None:
    with pytest.raises(ValueError, match="at least one metric"):
        context.profile(Mock(), config=LaunchConfig(), metrics=set())


def test_profiling_rejects_string_metrics(context: DeviceContext) -> None:
    with pytest.raises(TypeError, match="KernelMetric"):
        context.profile(Mock(), config=LaunchConfig(), metrics={"duration"})
