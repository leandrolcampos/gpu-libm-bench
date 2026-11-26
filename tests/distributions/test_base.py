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

"""Tests for the distribution interface."""

import numpy as np
import pytest

from gpu_libm_bench.distributions import Distribution, Uniform


def test_factory_constructs_uniform() -> None:
    distribution = Distribution.from_config(
        {"distribution": "uniform", "low": -10.0, "high": 10.0}, dtype=np.float32
    )

    assert isinstance(distribution, Uniform)
    assert distribution.low == -10.0
    assert distribution.high == 10.0
    assert distribution.dtype == np.dtype(np.float32)


def test_unknown_distribution_is_rejected() -> None:
    with pytest.raises(ValueError):
        Distribution.from_config({"distribution": "missing"}, dtype=np.float32)
