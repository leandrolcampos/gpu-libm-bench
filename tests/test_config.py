# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Tests for build and experiment configuration validation."""

from dataclasses import dataclass

import pytest
from hydra import compose, initialize_config_module
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from gpu_libm_bench.configs import (
    ArgumentConfig,
    MetricConfig,
    register_config_schema,
    validate_build_config,
    validate_experiment_config,
)


@pytest.fixture
def build_config() -> DictConfig:
    register_config_schema()
    with initialize_config_module(config_module="gpu_libm_bench.configs.yaml"):
        return compose(
            config_name="build_config",
            overrides=["backend=cuda", "library=llvm-libm"],
        )


@pytest.fixture
def experiment_config() -> DictConfig:
    register_config_schema()
    with initialize_config_module(config_module="gpu_libm_bench.configs.yaml"):
        return compose(
            config_name="experiment_config",
            overrides=[
                "function=expf",
                "metric=latency",
                "backend=cuda",
                "run.libraries=[llvm-libm]",
            ],
        )


def test_build_configuration_allows_no_library(build_config: DictConfig) -> None:
    build_config.library = None
    assert validate_build_config(build_config).library is None


@pytest.mark.parametrize("field", ["backend.name", "library.name"])
def test_build_rejects_empty_names(build_config: DictConfig, field: str) -> None:
    OmegaConf.update(build_config, field, " ")
    with pytest.raises(ValueError, match=f"{field} must not be empty"):
        validate_build_config(build_config)


def test_build_library_availability_is_not_checked_during_config_validation(
    build_config: DictConfig,
) -> None:
    build_config.library.backends = {}

    config = validate_build_config(build_config)

    assert config.library is not None
    assert config.library.backends == {}


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
            config_name="experiment_config",
            overrides=[
                "function=expf",
                "metric=_test_custom",
                "backend=cuda",
                "run.libraries=[llvm-libm]",
                "metric.batch_size=64",
            ],
        )

    metric = validate_experiment_config(config).metric
    assert isinstance(metric, CustomMetricConfig)
    assert metric.batch_size == 64


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("backend.name", "backend.name must not be empty"),
        ("libraries.llvm-libm.name", "library.name must not be empty"),
    ],
)
def test_experiment_rejects_empty_names(
    experiment_config: DictConfig, field: str, message: str
) -> None:
    OmegaConf.update(experiment_config, field, " ")
    with pytest.raises(ValueError, match=message):
        validate_experiment_config(experiment_config)


def test_experiment_requires_input_domains(experiment_config: DictConfig) -> None:
    experiment_config.function.domains = {}
    with pytest.raises(ValueError, match="function.domains"):
        validate_experiment_config(experiment_config)


@pytest.mark.parametrize("domains", [[], ["missing"], ["central", "central"]])
def test_invalid_domain_selection(
    experiment_config: DictConfig, domains: list[str]
) -> None:
    experiment_config.run.domains = domains
    with pytest.raises(ValueError, match="run.domains"):
        validate_experiment_config(experiment_config)


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
        validate_experiment_config(experiment_config)


def test_signature_requires_an_input_argument(experiment_config: DictConfig) -> None:
    experiment_config.function.signature.arguments[0].role = "output"
    with pytest.raises(ValueError, match="at least one input"):
        validate_experiment_config(experiment_config)


@pytest.mark.parametrize(
    ("names", "valid"),
    [([], False), (["x", "out"], False), (["x"], True)],
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
        validate_experiment_config(experiment_config)
    else:
        with pytest.raises(ValueError, match="domain 'wide'.*input signature"):
            validate_experiment_config(experiment_config)


def test_library_catalog_names_must_be_consistent(
    experiment_config: DictConfig,
) -> None:
    experiment_config.libraries["llvm-libm"].name = "other"
    with pytest.raises(ValueError, match="catalog keys"):
        validate_experiment_config(experiment_config)


@pytest.mark.parametrize("selection", [[], ["missing"], ["llvm-libm", "llvm-libm"]])
def test_invalid_library_selection(
    experiment_config: DictConfig, selection: list[str]
) -> None:
    experiment_config.run.libraries = selection
    with pytest.raises(ValueError, match="run.libraries"):
        validate_experiment_config(experiment_config)


def test_library_availability_is_not_checked_during_config_validation(
    experiment_config: DictConfig,
) -> None:
    experiment_config.libraries["llvm-libm"].backends = {}

    config = validate_experiment_config(experiment_config)

    assert config.libraries["llvm-libm"].backends == {}
