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

"""Tests for device duration conversion."""

import numpy as np
import pytest

from gpu_libm_bench.devices import TimerInfo


@pytest.mark.parametrize("frequency_hz", [1_000_000_000, 250_000_000, 3_000_000_000])
def test_ticks_are_converted_using_the_timer_frequency(frequency_hz: int) -> None:
    timer_info = TimerInfo("test counter", 64, frequency_hz)
    ticks = np.array([[0, 1, 2], [1234, 1_000_000, 9_000_000_000]], dtype=np.uint64)

    elapsed = timer_info.to_ns(ticks)

    expected = [
        [float(value) * 1_000_000_000 / frequency_hz for value in row]
        for row in ticks.tolist()
    ]
    np.testing.assert_allclose(elapsed, expected, rtol=1e-15)
    assert elapsed.dtype == np.dtype(np.float64)
