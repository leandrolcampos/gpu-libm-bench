# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Tests for experiment configuration."""

from dataclasses import dataclass

import pytest
from hydra import compose, initialize_config_module
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from gpu_libm_bench.configs import (
    ArgumentConfig,
    MetricConfig,
    register_config_schema,
    validate_config,
)


@pytest.fixture
def experiment_config() -> DictConfig:
    register_config_schema()
    with initialize_config_module(config_module="gpu_libm_bench.configs.yaml"):
        return compose(
            config_name="config",
            overrides=[
                "function=expf",
                "metric=latency",
                "library=llvm-libm",
                "backend=cuda",
            ],
        )


@pytest.mark.parametrize("domains", [[], ["missing"], ["central", "central"]])
def test_invalid_domain_selection(
    experiment_config: DictConfig, domains: list[str]
) -> None:
    experiment_config.run.domains = domains
    with pytest.raises(ValueError, match="run.domains"):
        validate_config(experiment_config)


def test_experiment_requires_input_domains(experiment_config: DictConfig) -> None:
    experiment_config.function.domains = {}
    with pytest.raises(ValueError, match="function.domains"):
        validate_config(experiment_config)


@pytest.mark.parametrize(
    ("names", "valid"),
    [([], False), (["x", "extra"], False), (["x", "out"], False), (["x"], True)],
)
def test_domain_arguments_must_match_only_signature_inputs(
    experiment_config: DictConfig, names: list[str], valid: bool
) -> None:
    experiment_config.function.signature.arguments.append(
        ArgumentConfig(name="out", type="float*", role="output")
    )
    experiment_config.function.domains.wide.generator.arguments = {
        name: {"distribution": "uniform", "low": -1, "high": 1} for name in names
    }

    if valid:
        validate_config(experiment_config)
    else:
        with pytest.raises(ValueError, match="domain 'wide'.*input signature"):
            validate_config(experiment_config)


@pytest.mark.parametrize(
    ("argument", "message"),
    [
        (ArgumentConfig(name="x", type="float", role="input"), "unique"),
        (ArgumentConfig(name="y", type="float", role="inout"), "roles"),
    ],
)
def test_signature_arguments_must_have_unique_names_and_valid_roles(
    experiment_config: DictConfig, argument: ArgumentConfig, message: str
) -> None:
    experiment_config.function.signature.arguments.append(argument)
    with pytest.raises(ValueError, match=message):
        validate_config(experiment_config)


def test_signature_requires_an_input_argument(experiment_config: DictConfig) -> None:
    experiment_config.function.signature.arguments[0].role = "output"
    with pytest.raises(ValueError, match="at least one input"):
        validate_config(experiment_config)


def test_library_must_support_selected_backend(experiment_config: DictConfig) -> None:
    experiment_config.library.backends = {}
    with pytest.raises(ValueError, match="does not declare support for backend 'cuda'"):
        validate_config(experiment_config)


def test_other_metrics_use_their_own_schema() -> None:
    @dataclass
    class CustomMetricConfig(MetricConfig):
        batch_size: int

    register_config_schema()
    ConfigStore.instance().store(
        group="metric",
        name="_test_custom",
        node=CustomMetricConfig(name="custom", unit="calls", batch_size=32),
        replace=True,
    )
    with initialize_config_module(config_module="gpu_libm_bench.configs.yaml"):
        config = compose(
            config_name="config",
            overrides=[
                "function=expf",
                "metric=_test_custom",
                "library=llvm-libm",
                "backend=cuda",
                "metric.batch_size=64",
            ],
        )

    metric = validate_config(config).metric
    assert isinstance(metric, CustomMetricConfig)
    assert metric.batch_size == 64
