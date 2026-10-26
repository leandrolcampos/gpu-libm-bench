# Copyright 2026 Leandro A. Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Tests for experiment input preparation and reproducible stream allocation."""

import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_module

from gpu_libm_bench.config import (
    DomainConfig,
    ExperimentConfig,
    LatencyConfig,
    RandomGeneratorConfig,
    register_config_schema,
    validate_config,
)
from gpu_libm_bench.experiment import Experiment


@pytest.fixture
def config() -> ExperimentConfig:
    register_config_schema()
    with initialize_config_module(config_module="gpu_libm_bench.conf"):
        composed = compose(
            config_name="config",
            overrides=[
                "function=expf",
                "metric=latency",
                "library=llvm-libm",
                "backend=cuda",
                "metric.sample_size=128",
            ],
        )
    return validate_config(composed)


def test_generate_expf_batches(config: ExperimentConfig) -> None:
    config = replace(
        config,
        function=replace(
            config.function,
            domains={
                **config.function.domains,
                "wide": DomainConfig(
                    generator=RandomGeneratorConfig(
                        name="random",
                        arguments={
                            "x": {"distribution": "uniform", "low": 20, "high": 30}
                        },
                    )
                ),
            },
        ),
    )
    assert isinstance(config.metric, LatencyConfig)
    batches = list(Experiment(config).generate_inputs(config.metric.sample_size))

    assert [batch.domain for batch in batches] == ["central", "wide"]
    for batch in batches:
        assert set(batch.arguments) == {"x"}
        samples = batch.arguments["x"]
        generator = config.function.domains[batch.domain].generator
        assert isinstance(generator, RandomGeneratorConfig)
        bounds = generator.arguments["x"]
        assert samples.dtype == np.dtype(np.float32)
        assert samples.shape == (config.metric.sample_size,)
        assert samples.flags.c_contiguous
        assert np.all(samples >= bounds["low"])
        assert np.all(samples <= bounds["high"])


def test_streams_are_reproducible_and_domain_local(config: ExperimentConfig) -> None:
    experiment = Experiment(config)
    first = {
        batch.domain: batch.arguments["x"] for batch in experiment.generate_inputs(32)
    }
    subsequent = next(experiment.generate_inputs(32))
    assert not np.array_equal(subsequent.arguments["x"], first[subsequent.domain])

    for domains in (None, ["wide"], ["wide", "central"]):
        selected = replace(config, run=replace(config.run, domains=domains))
        for batch in Experiment(selected).generate_inputs(32):
            np.testing.assert_array_equal(
                batch.arguments["x"].view(np.uint32),
                first[batch.domain].view(np.uint32),
            )

    extended = replace(
        config,
        function=replace(
            config.function,
            domains={
                "additional": config.function.domains["central"],
                **dict(reversed(list(config.function.domains.items()))),
            },
        ),
    )
    for batch in Experiment(extended).generate_inputs(32):
        if batch.domain in first:
            np.testing.assert_array_equal(
                batch.arguments["x"].view(np.uint32),
                first[batch.domain].view(np.uint32),
            )

    changed_seed = replace(config, run=replace(config.run, seed=123))
    changed = next(Experiment(changed_seed).generate_inputs(32))
    assert not np.array_equal(changed.arguments["x"], first[changed.domain])


def test_identical_domains_use_distinct_streams(config: ExperimentConfig) -> None:
    paired = replace(
        config,
        function=replace(
            config.function,
            domains={
                "left": config.function.domains["central"],
                "right": config.function.domains["central"],
            },
        ),
    )
    left, right = Experiment(paired).generate_inputs(32)
    assert not np.array_equal(left.arguments["x"], right.arguments["x"])


def test_input_sampling_does_not_consume_other_streams(
    config: ExperimentConfig,
) -> None:
    expected_ordering = Experiment(config).ordering_rng.random(16)
    expected_bootstrap = Experiment(config).bootstrap_rng.random(16)
    experiment = Experiment(config)

    list(experiment.generate_inputs(128))

    np.testing.assert_array_equal(experiment.ordering_rng.random(16), expected_ordering)
    np.testing.assert_array_equal(
        experiment.bootstrap_rng.random(16), expected_bootstrap
    )
    assert not np.array_equal(expected_ordering, expected_bootstrap)


def test_seed_policy_is_stable_across_processes(tmp_path: Path) -> None:
    script = """
from hydra import compose, initialize_config_module
from gpu_libm_bench.config import register_config_schema, validate_config
from gpu_libm_bench.experiment import Experiment

register_config_schema()
with initialize_config_module(config_module="gpu_libm_bench.conf"):
    config = compose(config_name="config", overrides=[
        "function=expf", "metric=latency", "library=llvm-libm", "backend=cuda",
    ])
batch = next(Experiment(validate_config(config)).generate_inputs(32))
print(batch.arguments["x"].tobytes().hex())
"""
    outputs = []
    for hash_seed in ("1", "2"):
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=tmp_path,
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        outputs.append(result.stdout)
    assert outputs[0] == outputs[1]
