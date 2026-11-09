# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Tests for finite, reproducible input generation and lazy batching."""

from unittest.mock import Mock

import numpy as np
import pytest

from gpu_libm_bench import inputs
from gpu_libm_bench.configs import (
    ArgumentConfig,
    RandomGeneratorConfig,
    SignatureConfig,
)
from gpu_libm_bench.distributions import Uniform

SIGNATURE = SignatureConfig("float", [ArgumentConfig("x", "float", "input")])
ARGUMENTS = {"x": {"distribution": "uniform", "low": -1, "high": 1}}
GENERATOR_CONFIG = RandomGeneratorConfig(name="random", arguments=ARGUMENTS)


def test_random_generator_uses_distinct_argument_streams() -> None:
    signature = SignatureConfig(
        "void",
        [
            ArgumentConfig("x", "float", "input"),
            ArgumentConfig("y", "float", "input"),
            ArgumentConfig("out", "double*", "output"),
        ],
    )
    generator = inputs.InputGenerator.from_config(
        RandomGeneratorConfig(
            name="random", arguments={"x": ARGUMENTS["x"], "y": ARGUMENTS["x"]}
        ),
        signature,
        sample_size=32,
        domain="paired",
        rng=np.random.default_rng(42),
    )

    batch = next(generator.batches(batch_size=32))

    assert batch.domain == "paired"
    assert set(batch.arguments) == {"x", "y"}
    assert not np.array_equal(batch.arguments["x"], batch.arguments["y"])


def test_factory_rejects_unknown_arguments() -> None:
    with pytest.raises(TypeError, match="unexpected keyword argument 'unknown'"):
        inputs.InputGenerator.from_config(
            GENERATOR_CONFIG,
            SIGNATURE,
            sample_size=5,
            domain="central",
            rng=np.random.default_rng(42),
            unknown=True,
        )


@pytest.mark.parametrize(
    ("sample_size", "error"), [(0, ValueError), (True, TypeError), (1.5, TypeError)]
)
def test_invalid_sample_size_is_rejected(
    sample_size: int | float, error: type[Exception]
) -> None:
    with pytest.raises(error, match="sample_size"):
        inputs.RandomGenerator(
            {"x": Uniform(low=-1, high=1, dtype=np.float32)},
            sample_size=sample_size,
            domain="central",
            rng=np.random.default_rng(42),
        )


def test_random_generator_requires_numpy_rng() -> None:
    with pytest.raises(TypeError, match="rng must be a NumPy Generator"):
        inputs.InputGenerator.from_config(
            GENERATOR_CONFIG, SIGNATURE, sample_size=5, domain="central", rng=None
        )


def test_generator_description_preserves_argument_order_without_consuming_rng() -> None:
    distributions = {
        "x": Uniform(low=-1, high=1, dtype=np.float32),
        "y": Uniform(low=2, high=3, dtype=np.float64),
    }
    generator = inputs.RandomGenerator(
        distributions, sample_size=16, domain="paired", rng=np.random.default_rng(42)
    )
    reference = inputs.RandomGenerator(
        distributions, sample_size=16, domain="paired", rng=np.random.default_rng(42)
    )

    description = str(generator)
    assert description == "x: uniform(-1, 1)\ny: uniform(2, 3)"

    actual = next(generator.batches(16))
    expected = next(reference.batches(16))

    for name in distributions:
        np.testing.assert_array_equal(actual.arguments[name], expected.arguments[name])


@pytest.mark.parametrize(
    ("batch_size", "error"), [(0, ValueError), (True, TypeError), (1.5, TypeError)]
)
def test_invalid_batch_size_is_rejected(
    batch_size: int | float, error: type[Exception]
) -> None:
    generator = inputs.InputGenerator.from_config(
        GENERATOR_CONFIG,
        SIGNATURE,
        sample_size=5,
        domain="central",
        rng=np.random.default_rng(42),
    )

    with pytest.raises(error, match="batch_size"):
        next(generator.batches(batch_size))


@pytest.mark.parametrize(
    ("batch_size", "drop_remainder", "expected_sizes"),
    [
        (None, False, [1, 1, 1, 1, 1]),
        (None, True, [1, 1, 1, 1, 1]),
        (3, False, [3, 2]),
        (3, True, [3]),
        (5, True, [5]),
        (8, False, [5]),
        (8, True, []),
    ],
)
def test_batch_iteration_is_finite_and_reproducible(
    batch_size: int | None, drop_remainder: bool, expected_sizes: list[int]
) -> None:
    generator = inputs.InputGenerator.from_config(
        GENERATOR_CONFIG,
        SIGNATURE,
        sample_size=5,
        domain="central",
        rng=np.random.default_rng(42),
    )

    first = list(generator.batches(batch_size, drop_remainder=drop_remainder))
    second = list(generator.batches(batch_size, drop_remainder=drop_remainder))

    assert [batch.size for batch in first] == expected_sizes
    assert [batch.size for batch in second] == expected_sizes
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left.arguments["x"], right.arguments["x"])


def test_batch_iterators_are_lazy_and_independent() -> None:
    distribution = Mock(wraps=Uniform(low=-1, high=1, dtype=np.float32))
    generator = inputs.RandomGenerator(
        {"x": distribution},
        sample_size=5,
        domain="central",
        rng=np.random.default_rng(42),
    )
    left = generator.batches(2)
    right = generator.batches(2)
    distribution.assert_not_called()

    first = next(left)
    second = next(left)
    repeated = next(right)

    np.testing.assert_array_equal(first.arguments["x"], repeated.arguments["x"])
    assert not np.array_equal(first.arguments["x"], second.arguments["x"])
    assert [batch.size for batch in left] == [1]
    assert [batch.size for batch in right] == [2, 1]
