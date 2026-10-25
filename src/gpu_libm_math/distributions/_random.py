# Copyright 2026 Leandro A. Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Unit-interval random sampling."""

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from gpu_libm_math.types import FloatDType, FloatScalar, SampleShape


def random(
    rng: Generator, sample_size: SampleShape, dtype: FloatDType
) -> NDArray[FloatScalar]:
    if dtype == np.dtype(np.float16):
        significand_bits = 11
        samples = rng.integers(
            0, 1 << significand_bits, size=sample_size, dtype=np.uint16
        ).astype(np.float16)
        samples *= np.float16(2.0**-significand_bits)
        return samples

    if dtype == np.dtype(np.float32):
        return rng.random(size=sample_size, dtype=np.float32)

    return rng.random(size=sample_size, dtype=np.float64)
