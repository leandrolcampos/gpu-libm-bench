# Copyright 2026 Leandro A. Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Tests for input generation and batching."""

import numpy as np
import pytest

from gpu_libm_bench import inputs
from gpu_libm_bench.config import (
    ArgumentConfig,
    RandomGeneratorConfig,
    SignatureConfig,
)
from gpu_libm_bench.distributions import Uniform

SIGNATURE = SignatureConfig("float", [ArgumentConfig("x", "float", "input")])
ARGUMENTS = {"x": {"distribution": "uniform", "low": -1, "high": 1}}
GENERATOR_CONFIG = RandomGeneratorConfig(name="random", arguments=ARGUMENTS)


def test_generator_is_consumed_across_arguments_and_calls() -> None:
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
        domain_name="paired",
        rng=np.random.default_rng(42),
    )
    first = next(generator.batches(batch_size=32)).arguments
    second = generator.generate(32).arguments

    assert set(first) == {"x", "y"}
    assert not np.array_equal(first["x"], first["y"])
    assert not np.array_equal(first["x"], second["x"])


@pytest.mark.parametrize("batch_size", [0, -1])
def test_batch_size_must_be_positive(batch_size: int) -> None:
    generator = inputs.InputGenerator.from_config(
        GENERATOR_CONFIG,
        SIGNATURE,
        domain_name="central",
        rng=np.random.default_rng(42),
    )
    with pytest.raises(ValueError, match="batch_size"):
        generator.generate(batch_size)


def test_random_generator_requires_explicit_rng() -> None:
    with pytest.raises(ValueError, match="explicit Generator"):
        inputs.InputGenerator.from_config(
            GENERATOR_CONFIG, SIGNATURE, domain_name="central"
        )


def test_argument_streams_survive_batch_partition_and_mapping_reordering() -> None:
    distributions = {
        "z": Uniform(low=-1, high=1, dtype=np.float16),
        "x": Uniform(low=-1, high=1, dtype=np.float32),
        "y": Uniform(low=-1, high=1, dtype=np.float64),
    }
    whole = inputs.RandomGenerator(
        distributions, np.random.default_rng(42), domain="paired"
    ).generate(17)
    generator = inputs.RandomGenerator(
        dict(reversed(list(distributions.items()))),
        np.random.default_rng(42),
        domain="paired",
    )
    parts = [generator.generate(size) for size in (3, 5, 9)]

    for name, samples in whole.arguments.items():
        concatenated = np.concatenate([part.arguments[name] for part in parts])
        np.testing.assert_array_equal(
            samples.view(np.uint8), concatenated.view(np.uint8)
        )


@pytest.mark.parametrize(
    ("batch_size", "options", "expected_sizes"),
    [
        (3, {}, [3, 2]),
        (3, {"drop_remainder": True}, [3]),
        (5, {"drop_remainder": True}, [5]),
    ],
)
def test_batch_iteration_handles_remainder_and_exhaustion(
    batch_size: int, options: dict[str, bool], expected_sizes: list[int]
) -> None:
    class FiniteGenerator(inputs.InputGenerator):
        def __init__(self) -> None:
            self.position = 0

        def generate(self, batch_size: int) -> inputs.InputBatch:
            if self.position == 5:
                raise StopIteration
            start = self.position
            self.position = min(start + batch_size, 5)
            return inputs.InputBatch(
                "finite", {"x": np.arange(start, self.position, dtype=np.float32)}
            )

    generator = FiniteGenerator()
    batches = list(generator.batches(batch_size=batch_size, **options))

    assert [batch.size for batch in batches] == expected_sizes
    np.testing.assert_array_equal(
        np.concatenate([batch.arguments["x"] for batch in batches]),
        np.arange(sum(expected_sizes)),
    )
    assert list(generator.batches(batch_size=batch_size, **options)) == []
