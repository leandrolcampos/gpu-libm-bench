# Copyright 2026 Leandro A. Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Uniform input distribution."""

from dataclasses import dataclass, field
from math import isfinite
from numbers import Real
from typing import cast, overload

import numpy as np
from numpy.typing import DTypeLike, NDArray

from gpu_libm_bench.types import FloatDType, FloatScalar, SampleShape, SampleSize

from ._random import random
from .base import Distribution


@dataclass(frozen=True, slots=True, init=False)
class Uniform(Distribution):
    low: float
    high: float
    dtype: FloatDType
    _low_value: FloatScalar = field(repr=False, compare=False)
    _high_value: FloatScalar = field(repr=False, compare=False)

    def __init__(self, *, low: float, high: float, dtype: DTypeLike) -> None:
        if (
            not isinstance(low, Real)
            or not isinstance(high, Real)
            or isinstance(low, (bool, np.bool_))
            or isinstance(high, (bool, np.bool_))
        ):
            raise TypeError("uniform bounds must be real numbers")

        low, high = float(low), float(high)
        if not isfinite(low) or not isfinite(high):
            raise ValueError("uniform bounds must be finite")
        if low >= high:
            raise ValueError("uniform requires low < high")

        normalized_dtype = np.dtype(dtype)
        if normalized_dtype not in (
            np.dtype(np.float16),
            np.dtype(np.float32),
            np.dtype(np.float64),
        ):
            raise TypeError("uniform requires native float16, float32 or float64 dtype")
        normalized_dtype = cast("FloatDType", normalized_dtype)

        with np.errstate(over="ignore", under="ignore"):
            low_value = normalized_dtype.type(low)
            high_value = normalized_dtype.type(high)
            if not np.isfinite(low_value) or not np.isfinite(high_value):
                raise ValueError("uniform bounds must round to finite dtype values")

            if float(low_value) < low:
                low_value = np.nextafter(low_value, normalized_dtype.type(np.inf))

            if float(high_value) > high:
                high_value = np.nextafter(high_value, normalized_dtype.type(-np.inf))

        if (
            not np.isfinite(low_value)
            or not np.isfinite(high_value)
            or low_value > high_value
        ):
            raise ValueError("uniform interval contains no representable dtype values")

        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)
        object.__setattr__(self, "dtype", normalized_dtype)
        object.__setattr__(self, "_low_value", low_value)
        object.__setattr__(self, "_high_value", high_value)

    @overload
    def __call__(
        self, rng: np.random.Generator, sample_size: None = None
    ) -> FloatScalar: ...

    @overload
    def __call__(
        self, rng: np.random.Generator, sample_size: SampleShape
    ) -> NDArray[FloatScalar]: ...

    @overload
    def __call__(
        self, rng: np.random.Generator, sample_size: SampleSize = None
    ) -> FloatScalar | NDArray[FloatScalar]: ...

    def __call__(
        self,
        rng: np.random.Generator,
        sample_size: SampleSize = None,
    ) -> FloatScalar | NDArray[FloatScalar]:
        # NumPy ignores dtype for size=None; an empty shape preserves it.
        shape = () if sample_size is None else sample_size
        samples = random(rng, shape, self.dtype)

        if self._low_value < 0 < self._high_value:
            # A convex combination avoids overflow in high - low across zero.
            one = self.dtype.type(1)
            complement = np.subtract(one, samples)
            samples *= self._high_value
            np.add(samples, complement * self._low_value, out=samples)
        else:
            samples *= self._high_value - self._low_value
            samples += self._low_value

        if sample_size is None:
            return cast("FloatScalar", samples[()])
        return samples

    def __str__(self) -> str:
        low = str(self.low).removesuffix(".0")
        high = str(self.high).removesuffix(".0")
        return f"uniform({low}, {high})"
