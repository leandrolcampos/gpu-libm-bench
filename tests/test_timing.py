# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Tests for device duration conversion."""

import numpy as np
import pytest

from gpu_libm_bench.devices import TimerInfo


@pytest.mark.parametrize("frequency_hz", [1_000_000_000, 250_000_000, 3_000_000_000])
def test_ticks_are_converted_using_the_timer_frequency(frequency_hz: int) -> None:
    timer = TimerInfo("test counter", 64, frequency_hz)
    ticks = np.array([[0, 1, 2], [1234, 1_000_000, 9_000_000_000]], dtype=np.uint64)

    elapsed = timer.to_nanoseconds(ticks)

    expected = [
        [float(value) * 1_000_000_000 / frequency_hz for value in row]
        for row in ticks.tolist()
    ]
    np.testing.assert_allclose(elapsed, expected, rtol=1e-15)
    assert elapsed.dtype == np.dtype(np.float64)
