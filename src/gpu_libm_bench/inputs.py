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

"""Input generators and batches for mathematical functions."""

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass
from operator import index
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .configs import (
    DistributionConfig,
    InputGeneratorConfig,
    RandomGeneratorConfig,
    SignatureConfig,
)
from .distributions import Distribution
from .types import FloatType, Scalar

_INPUT_DTYPES: dict[str, FloatType] = {
    "float": np.float32,
    "double": np.float64,
    "_Float16": np.float16,
}


@dataclass(frozen=True, slots=True)
class InputBatch:
    domain: str
    arguments: dict[str, NDArray[Scalar]]

    @property
    def size(self) -> int:
        return len(next(iter(self.arguments.values())))


class InputGenerator(ABC):
    @staticmethod
    def from_config(
        config: InputGeneratorConfig,
        signature: SignatureConfig,
        **kwargs: Any,
    ) -> "InputGenerator":
        if config.name == "random":
            return RandomGenerator.from_config(config, signature, **kwargs)
        raise ValueError(f"unknown input generator: {config.name!r}")

    @abstractmethod
    def __str__(self) -> str: ...

    @abstractmethod
    def batches(
        self,
        batch_size: int | None = None,
        *,
        batch_count: int | None = None,
        drop_remainder: bool = False,
    ) -> Iterator[InputBatch]: ...


class RandomGenerator(InputGenerator):
    def __init__(
        self,
        distributions: Mapping[str, Distribution[Scalar]],
        *,
        sample_size: int | None,
        domain: str,
        rng: np.random.Generator,
    ) -> None:
        if not distributions:
            raise ValueError("at least one input distribution is required")

        if sample_size is not None:
            sample_size = _normalize_positive_integer(sample_size, "sample_size")

        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a NumPy Generator")

        self._distributions = dict(distributions)
        names = sorted(self._distributions)
        self._rngs = dict(zip(names, rng.spawn(len(names)), strict=True))
        self._sample_size = sample_size
        self._domain = domain

    @staticmethod
    def from_config(
        config: InputGeneratorConfig,
        signature: SignatureConfig,
        **kwargs: Any,
    ) -> "RandomGenerator":
        if not isinstance(config, RandomGeneratorConfig):
            raise ValueError("random input generation requires RandomGeneratorConfig")

        distributions = _distributions_from_config(signature, config.arguments)
        return RandomGenerator(distributions, **kwargs)

    def __str__(self) -> str:
        return "\n".join(
            f"{name}: {distribution}"
            for name, distribution in self._distributions.items()
        )

    def batches(
        self,
        batch_size: int | None = None,
        *,
        batch_count: int | None = None,
        drop_remainder: bool = False,
    ) -> Iterator[InputBatch]:
        batch_size = (
            1
            if batch_size is None
            else _normalize_positive_integer(batch_size, "batch_size")
        )
        sample_size = self._sample_size
        if batch_count is not None:
            sample_size = batch_size * _normalize_positive_integer(
                batch_count, "batch_count"
            )
        if sample_size is None:
            raise ValueError("batch_count is required when sample_size is unspecified")

        rngs = deepcopy(self._rngs)
        for start in range(0, sample_size, batch_size):
            size = min(batch_size, sample_size - start)
            if drop_remainder and size < batch_size:
                return

            yield InputBatch(
                domain=self._domain,
                arguments={
                    name: distribution(rngs[name], size)
                    for name, distribution in self._distributions.items()
                },
            )


def _distributions_from_config(
    signature: SignatureConfig, parameters: Mapping[str, DistributionConfig]
) -> dict[str, Distribution[Scalar]]:
    arguments = [
        argument for argument in signature.arguments if argument.role == "input"
    ]

    distributions: dict[str, Distribution[Scalar]] = {}
    for argument in arguments:
        try:
            dtype = _INPUT_DTYPES[argument.type]
        except KeyError as error:
            raise ValueError(
                f"unsupported input type {argument.type!r} "
                f"for argument {argument.name!r}"
            ) from error

        try:
            distributions[argument.name] = Distribution.from_config(
                parameters[argument.name], dtype=dtype
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"argument {argument.name!r}: {error}") from error

    return distributions


def _normalize_positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    try:
        value = index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer") from None
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value
