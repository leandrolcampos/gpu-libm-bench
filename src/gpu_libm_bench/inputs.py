# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

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
        self, batch_size: int | None = None, *, drop_remainder: bool = False
    ) -> Iterator[InputBatch]: ...


class RandomGenerator(InputGenerator):
    def __init__(
        self,
        distributions: Mapping[str, Distribution[Scalar]],
        *,
        sample_size: int,
        domain: str,
        rng: np.random.Generator,
    ) -> None:
        if not distributions:
            raise ValueError("at least one input distribution is required")

        if isinstance(sample_size, (bool, np.bool_)):
            raise TypeError("sample_size must be an integer")
        try:
            sample_size = index(sample_size)
        except TypeError:
            raise TypeError("sample_size must be an integer") from None
        if sample_size <= 0:
            raise ValueError("sample_size must be positive")

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
        self, batch_size: int | None = None, *, drop_remainder: bool = False
    ) -> Iterator[InputBatch]:
        if batch_size is None:
            batch_size = 1
        if isinstance(batch_size, (bool, np.bool_)):
            raise TypeError("batch_size must be an integer")
        try:
            batch_size = index(batch_size)
        except TypeError:
            raise TypeError("batch_size must be an integer") from None
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        rngs = deepcopy(self._rngs)
        for start in range(0, self._sample_size, batch_size):
            size = min(batch_size, self._sample_size - start)
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
                f"unsupported input type {argument.type!r} for argument {argument.name!r}"
            ) from error

        try:
            distributions[argument.name] = Distribution.from_config(
                parameters[argument.name], dtype=dtype
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"argument {argument.name!r}: {error}") from error

    return distributions
