# Copyright 2026 Leandro A. Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Input distribution interface."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from gpu_libm_math.types import SampleSize


class Distribution(ABC):
    @staticmethod
    def create(name: str, **parameters: Any) -> "Distribution":
        if name == "uniform":
            from .uniform import Uniform

            return Uniform(**parameters)

        raise ValueError(f"Unknown distribution: {name!r}")

    @abstractmethod
    def __call__(
        self,
        rng: Generator,
        sample_size: SampleSize = None,
    ) -> np.generic | NDArray[np.generic]: ...

    @abstractmethod
    def __str__(self) -> str: ...
