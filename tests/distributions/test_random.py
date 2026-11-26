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

"""Tests for unit-interval random sampling."""

from unittest.mock import Mock

import numpy as np
import pytest

from gpu_libm_bench.distributions._random import random


def test_float16_grid_is_exact() -> None:
    rng = Mock(spec=np.random.Generator)
    rng.integers.return_value = np.arange(2048, dtype=np.uint16)

    samples = random(rng, 2048, np.dtype(np.float16))

    assert samples.dtype == np.dtype(np.float16)
    assert samples[0] == 0
    assert samples[-1] == 1 - 2**-11
    np.testing.assert_array_equal(
        np.diff(samples.astype(np.float64)), np.full(2047, 2**-11)
    )


def test_unsupported_dtype_is_rejected() -> None:
    with pytest.raises(TypeError, match="unsupported dtype"):
        random(np.random.default_rng(42), 8, np.dtype(np.int32))
