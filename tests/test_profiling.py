# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import pytest

from gpu_libm_bench.devices import DeviceContext, KernelMetric, LaunchConfig


@pytest.fixture
def context(monkeypatch: pytest.MonkeyPatch) -> DeviceContext:
    backend = Mock()
    monkeypatch.setattr(
        "gpu_libm_bench.devices.context._load_backend", lambda name: backend
    )
    return DeviceContext()


@pytest.mark.parametrize(
    "metrics",
    [
        {KernelMetric.AVERAGE_FREQUENCY},
        {KernelMetric.DURATION},
        {KernelMetric.AVERAGE_FREQUENCY, KernelMetric.DURATION},
    ],
)
def test_profile_is_not_implemented(
    context: DeviceContext, metrics: set[KernelMetric]
) -> None:
    with patch.object(context, "launch") as launch:
        with pytest.raises(NotImplementedError, match="kernel profiling"):
            context.profile(Mock(), config=LaunchConfig(), metrics=metrics)

        launch.assert_not_called()


def test_profiling_rejects_empty_metrics(context: DeviceContext) -> None:
    with pytest.raises(ValueError, match="at least one metric"):
        context.profile(Mock(), config=LaunchConfig(), metrics=set())


def test_profiling_rejects_string_metrics(context: DeviceContext) -> None:
    with pytest.raises(TypeError, match="KernelMetric"):
        context.profile(Mock(), config=LaunchConfig(), metrics={"duration"})
