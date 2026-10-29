# Copyright 2026 Leandro A. Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Interfaces and factories for input distributions."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, overload

import numpy as np
from numpy.typing import DTypeLike, NDArray

from gpu_libm_bench.types import SampleShape, SampleSize, Scalar


class Distribution[T: Scalar](ABC):
    @staticmethod
    @overload
    def from_config[S: Scalar](
        config: Mapping[str, Any], *, dtype: type[S] | np.dtype[S]
    ) -> "Distribution[S]": ...

    @staticmethod
    @overload
    def from_config(
        config: Mapping[str, Any], *, dtype: DTypeLike
    ) -> "Distribution[Any]": ...

    @staticmethod
    def from_config(
        config: Mapping[str, Any], *, dtype: DTypeLike
    ) -> "Distribution[Any]":
        parameters = dict(config)
        try:
            name = parameters.pop("distribution")
        except KeyError as error:
            raise ValueError(
                "a distribution configuration requires 'distribution'"
            ) from error
        if name == "uniform":
            from .uniform import Uniform

            return Uniform(dtype=dtype, **parameters)

        raise ValueError(f"unknown distribution: {name!r}")

    @overload
    def __call__(self, rng: np.random.Generator, sample_size: None = None) -> T: ...

    @overload
    def __call__(
        self, rng: np.random.Generator, sample_size: SampleShape
    ) -> NDArray[T]: ...

    @abstractmethod
    def __call__(
        self,
        rng: np.random.Generator,
        sample_size: SampleSize = None,
    ) -> T | NDArray[T]: ...

    @abstractmethod
    def __str__(self) -> str: ...
