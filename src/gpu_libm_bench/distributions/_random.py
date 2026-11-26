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

"""Floating-point sampling on the unit interval."""

from typing import cast

import numpy as np
from numpy.typing import NDArray

from gpu_libm_bench.types import FloatScalar, SampleShape


def random[T: FloatScalar](
    rng: np.random.Generator, sample_size: SampleShape, dtype: np.dtype[T]
) -> NDArray[T]:
    samples: NDArray[FloatScalar]
    if dtype == np.dtype(np.float16):
        significand_bits = 11
        # Use 32-bit draws to preserve the stream across odd-sized batches.
        samples = rng.integers(
            0, 1 << significand_bits, size=sample_size, dtype=np.uint32
        ).astype(np.float16)
        samples *= np.float16(2.0**-significand_bits)
    elif dtype == np.dtype(np.float32):
        samples = rng.random(size=sample_size, dtype=np.float32)
    elif dtype == np.dtype(np.float64):
        samples = rng.random(size=sample_size, dtype=np.float64)
    else:
        raise TypeError(
            f"unsupported dtype {dtype!r}; expected native float16, float32 or float64"
        )

    return cast("NDArray[T]", samples)
