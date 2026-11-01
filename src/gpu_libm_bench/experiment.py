# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Benchmark experiment orchestration."""

import hashlib
import json
from collections.abc import Iterator

import numpy as np
from numpy.random import SeedSequence

from . import inputs
from .configs import ExperimentConfig


class Experiment:
    def __init__(self, config: ExperimentConfig) -> None:
        inputs_seed, ordering_seed, bootstrap_seed = SeedSequence(
            config.run.seed
        ).spawn(3)
        self.ordering_rng = np.random.default_rng(ordering_seed)
        self.bootstrap_rng = np.random.default_rng(bootstrap_seed)
        input_entropy = inputs_seed.generate_state(4).tolist()

        function = config.function
        names = function.domains if config.run.domains is None else config.run.domains
        self._input_generators = {
            name: inputs.InputGenerator.from_config(
                function.domains[name].generator,
                function.signature,
                domain_name=name,
                rng=_domain_rng(input_entropy, function.name, name),
            )
            for name in names
        }

    def generate_inputs(self, sample_size: int) -> Iterator[inputs.InputBatch]:
        for generator in self._input_generators.values():
            yield generator.generate(sample_size)


def _domain_rng(
    input_entropy: list[int], function_name: str, domain_name: str
) -> np.random.Generator:
    label = json.dumps(
        ["gpu-libm-bench:inputs:v3", function_name, domain_name],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(label).digest()
    identity = [int.from_bytes(digest[i : i + 4], "little") for i in range(0, 32, 4)]
    return np.random.default_rng(SeedSequence([*identity, *input_entropy]))
