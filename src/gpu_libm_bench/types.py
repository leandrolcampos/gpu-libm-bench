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
