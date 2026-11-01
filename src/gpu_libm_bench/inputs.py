# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Input generators and batches for mathematical functions."""

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from operator import index

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
        *,
        domain_name: str,
        rng: np.random.Generator | None = None,
    ) -> "InputGenerator":
        if config.name != "random" or not isinstance(config, RandomGeneratorConfig):
            raise ValueError(f"unknown input generator: {config.name!r}")
        if rng is None:
            raise ValueError("random input generation requires an explicit Generator")

        distributions = _distributions_from_config(signature, config.arguments)
        return RandomGenerator(distributions, rng, domain=domain_name)

    @abstractmethod
    def generate(self, batch_size: int) -> InputBatch: ...

    def batches(
        self, batch_size: int, *, drop_remainder: bool = False
    ) -> Iterator[InputBatch]:
        while True:
            try:
                batch = self.generate(batch_size)
            except StopIteration:
                return

            if drop_remainder and batch.size < batch_size:
                return

            yield batch


class RandomGenerator(InputGenerator):
    def __init__(
        self,
        distributions: Mapping[str, Distribution[Scalar]],
        rng: np.random.Generator,
        *,
        domain: str,
    ) -> None:
        if not distributions:
            raise ValueError("at least one input distribution is required")

        self._distributions = dict(distributions)
        names = sorted(self._distributions)
        self._rngs = dict(zip(names, rng.spawn(len(names)), strict=True))
        self._domain = domain

    def generate(self, batch_size: int) -> InputBatch:
        if isinstance(batch_size, (bool, np.bool_)):
            raise TypeError("batch_size must be an integer")

        batch_size = index(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        return InputBatch(
            domain=self._domain,
            arguments={
                name: distribution(self._rngs[name], batch_size)
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
