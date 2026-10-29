# Copyright 2026 Leandro A. Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Interfaces and factories for input distributions."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import DTypeLike, NDArray

from gpu_libm_bench.types import SampleSize, Scalar


class Distribution(ABC):
    @staticmethod
    def from_config(config: Mapping[str, Any], *, dtype: DTypeLike) -> "Distribution":
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

    @abstractmethod
    def __call__(
        self,
        rng: np.random.Generator,
        sample_size: SampleSize = None,
    ) -> Scalar | NDArray[Scalar]: ...

    @abstractmethod
    def __str__(self) -> str: ...
