# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for the CUDA backend."""

import pytest

from gpu_libm_bench.devices import DeviceContext

pytestmark = pytest.mark.cuda


def test_query_visible_devices() -> None:
    count = DeviceContext.device_count()
    if count == 0:
        pytest.skip("No NVIDIA devices are visible to the CUDA driver.")

    for device_id in range(count):
        info = DeviceContext.get_device_info(device_id)

        assert info.backend == "cuda"
        assert info.id == device_id
        assert info.name
        assert info.architecture.startswith("sm_")
        assert info.total_memory_bytes > 0

        assert info.timer.source == "%globaltimer"
        assert info.timer.counter_bits == 64
        assert info.timer.frequency_hz == 1_000_000_000

        with DeviceContext(device_id) as context:
            cached_info = context.device_info
            assert cached_info == info
            assert context.device_info is cached_info


def test_negative_device_id_is_rejected() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        DeviceContext.get_device_info(-1)


def test_out_of_range_device_id_is_rejected() -> None:
    count = DeviceContext.device_count()
    with pytest.raises(ValueError, match="out of range"):
        DeviceContext.get_device_info(count)
