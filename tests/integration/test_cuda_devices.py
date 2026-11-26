# ===----------------------------------------------------------------------=== #
# Copyright 2026 Leandro Lacerda Campos
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Integration tests for the CUDA backend."""

import pytest

from gpu_libm_bench.devices import DeviceContext

pytestmark = pytest.mark.cuda


def test_query_visible_devices() -> None:
    count = DeviceContext.device_count()
    if count == 0:
        pytest.skip("No NVIDIA devices are visible to the CUDA driver.")

    for device_id in range(count):
        device_info = DeviceContext.get_device_info(device_id)

        assert device_info.backend == "cuda"
        assert device_info.id == device_id
        assert device_info.name
        assert device_info.architecture.startswith("sm_")
        assert device_info.multiprocessor_count > 0
        assert device_info.warp_size == 32
        assert device_info.default_clock_frequency_hz > 0
        assert device_info.total_memory_bytes > 0
        assert device_info.l2_cache_size_bytes >= 0

        launch_limits = device_info.launch_limits
        for dimensions in (launch_limits.max_grid_dim, launch_limits.max_block_dim):
            assert isinstance(dimensions, tuple)
            assert len(dimensions) == 3
            assert min(dimensions) > 0
        assert launch_limits.max_threads_per_block > 0
        assert launch_limits.max_shared_memory_bytes_per_block > 0

        assert device_info.timer_info.source == "%globaltimer"
        assert device_info.timer_info.counter_bits == 64
        assert device_info.timer_info.frequency_hz == 1_000_000_000

        with DeviceContext(device_id) as context:
            cached_device_info = context.device_info
            assert cached_device_info == device_info
            assert context.device_info is cached_device_info


def test_negative_device_id_is_rejected() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        DeviceContext.get_device_info(-1)


def test_out_of_range_device_id_is_rejected() -> None:
    count = DeviceContext.device_count()
    with pytest.raises(ValueError, match="out of range"):
        DeviceContext.get_device_info(count)
