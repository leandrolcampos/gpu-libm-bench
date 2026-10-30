# Copyright 2026 Leandro A. Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Tests for latency protocol parameters."""

from unittest.mock import Mock

import pytest

from gpu_libm_bench.devices import DeviceBuffer, DeviceContext, DeviceKernel
from gpu_libm_bench.metrics import LatencyBenchmark


@pytest.mark.parametrize("length", [0, -1, 2**32])
def test_chain_length_must_fit_a_positive_uint32(length: int) -> None:
    with pytest.raises(ValueError, match="chain_length must be between"):
        LatencyBenchmark(
            Mock(spec=DeviceContext), Mock(spec=DeviceKernel), chain_length=length
        )


def test_empty_input_is_rejected_before_launch() -> None:
    context = Mock(spec=DeviceContext)
    benchmark = LatencyBenchmark(context, Mock(spec=DeviceKernel), chain_length=1)

    with pytest.raises(ValueError, match="x must have capacity"):
        benchmark.run_chain(
            Mock(spec=DeviceBuffer, size=0),
            Mock(spec=DeviceBuffer, size=1),
            Mock(spec=DeviceBuffer, size=1),
        )

    context.launch.assert_not_called()
