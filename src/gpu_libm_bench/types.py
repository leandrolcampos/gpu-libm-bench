# Copyright 2026 Leandro A. Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Shared numeric and array types."""

from collections.abc import Sequence
from typing import Any, SupportsIndex

import numpy as np

type SampleShape = SupportsIndex | Sequence[SupportsIndex]
type SampleSize = SampleShape | None
type Scalar = np.bool_ | np.number[Any]
type FloatScalar = np.float16 | np.float32 | np.float64
type FloatDType = np.dtype[np.float16] | np.dtype[np.float32] | np.dtype[np.float64]
type FloatType = type[np.float16] | type[np.float32] | type[np.float64]
