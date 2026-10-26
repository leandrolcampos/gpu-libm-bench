# Copyright 2026 Leandro A. Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

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
    with pytest.raises(TypeError, match="Unsupported dtype"):
        random(np.random.default_rng(42), 8, np.dtype(np.int32))
