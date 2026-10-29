# Copyright 2026 Leandro A. Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Floating-point sampling on the unit interval."""

import numpy as np
from numpy.typing import NDArray

from gpu_libm_bench.types import FloatDType, FloatScalar, SampleShape


def random(
    rng: np.random.Generator, sample_size: SampleShape, dtype: FloatDType
) -> NDArray[FloatScalar]:
    if dtype == np.dtype(np.float16):
        significand_bits = 11
        # Use 32-bit draws to preserve the stream across odd-sized batches.
        samples = rng.integers(
            0, 1 << significand_bits, size=sample_size, dtype=np.uint32
        ).astype(np.float16)
        samples *= np.float16(2.0**-significand_bits)
        return samples

    if dtype == np.dtype(np.float32):
        return rng.random(size=sample_size, dtype=np.float32)

    if dtype == np.dtype(np.float64):
        return rng.random(size=sample_size, dtype=np.float64)

    raise TypeError(
        f"unsupported dtype {dtype!r}; expected native float16, float32 or float64"
    )
