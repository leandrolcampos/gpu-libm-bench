# Copyright 2026 Leandro A. Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Tests for the distribution interface."""

import numpy as np
import pytest

from gpu_libm_math.distributions import Distribution, Uniform


def test_factory_constructs_uniform() -> None:
    distribution = Distribution.create(
        "uniform", low=-10.0, high=10.0, dtype=np.float32
    )

    assert isinstance(distribution, Uniform)
    assert distribution.low == -10.0
    assert distribution.high == 10.0
    assert distribution.dtype == np.dtype(np.float32)


def test_unknown_distribution_is_rejected() -> None:
    with pytest.raises(ValueError):
        Distribution.create("missing")
