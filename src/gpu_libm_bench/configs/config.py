# Copyright 2026 Leandro A. Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Experiment configuration schemas and consistency validation."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, cast

from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf


@dataclass
class ArgumentConfig:
    name: str
    type: str
    role: str


@dataclass
class SignatureConfig:
    return_type: str
    arguments: list[ArgumentConfig]


DistributionConfig = dict[str, Any]


@dataclass
class InputGeneratorConfig:
    name: str


@dataclass
class RandomGeneratorConfig(InputGeneratorConfig):
    arguments: dict[str, DistributionConfig]


@dataclass
class DomainConfig:
    generator: InputGeneratorConfig


@dataclass
class FunctionConfig:
    name: str
    signature: SignatureConfig
    domains: dict[str, DomainConfig]


@dataclass
class MetricConfig:
    name: str
    unit: str


@dataclass
class LatencyConfig(MetricConfig):
    sample_size: int
    chain_length: int
    repetitions: int
    warmup_runs: int
    bootstrap_resamples: int
    confidence_level: float


@dataclass
class LibraryArtifactConfig:
    bitcode_path: str | None


@dataclass
class LibraryConfig:
    name: str
    backends: dict[str, LibraryArtifactConfig]


@dataclass
class BackendConfig:
    name: str
    target_triple: str
    architecture: str | None
    device_id: int
    sdk_root: str | None


@dataclass
class RunConfig:
    seed: int
    domains: list[str] | None


@dataclass
class LLVMConfig:
    root: str | None


@dataclass
class ExperimentConfig:
    function: FunctionConfig
    metric: MetricConfig
    library: LibraryConfig
    backend: BackendConfig
    run: RunConfig
    llvm: LLVMConfig


def register_config_schema() -> None:
    ConfigStore.instance().store(
        name="experiment_schema",
        node=ExperimentConfig,
        provider="gpu-libm-bench",
        replace=True,
    )
    ConfigStore.instance().store(
        group="schema/metric",
        name="latency",
        node=LatencyConfig,
        package="metric",
        provider="gpu-libm-bench",
        replace=True,
    )
    ConfigStore.instance().store(
        group="schema/input_generator",
        name="random",
        node=RandomGeneratorConfig,
        provider="gpu-libm-bench",
        replace=True,
    )


def validate_config(config: DictConfig) -> ExperimentConfig:
    structured = OmegaConf.merge(OmegaConf.structured(ExperimentConfig), config)
    experiment = cast("ExperimentConfig", OmegaConf.to_object(structured))

    domains = experiment.function.domains
    if not domains:
        raise ValueError("function.domains must contain at least one domain")
    if any(not name.strip() for name in domains):
        raise ValueError("domain names must not be empty")

    selected = experiment.run.domains
    if selected is not None:
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("run.domains must be a nonempty list without duplicates")
        for name in selected:
            if name not in domains:
                raise ValueError(f"run.domains contains an unknown domain: {name!r}")

    _validate_function_inputs(
        experiment.function, domains if selected is None else selected
    )

    if experiment.backend.name not in experiment.library.backends:
        raise ValueError(
            f"library {experiment.library.name!r} does not declare support for "
            f"backend {experiment.backend.name!r}"
        )
    return experiment


def _validate_function_inputs(
    function: FunctionConfig, domain_names: Iterable[str]
) -> None:
    arguments = function.signature.arguments
    names = [argument.name for argument in arguments]
    if len(set(names)) != len(names):
        raise ValueError("signature argument names must be unique")
    if any(argument.role not in {"input", "output"} for argument in arguments):
        raise ValueError("signature argument roles must be 'input' or 'output'")
    inputs = {argument.name for argument in arguments if argument.role == "input"}
    if not inputs:
        raise ValueError("signature must declare at least one input argument")

    for name in domain_names:
        generator = function.domains[name].generator
        if (
            isinstance(generator, RandomGeneratorConfig)
            and set(generator.arguments) != inputs
        ):
            raise ValueError(
                f"domain {name!r} arguments must match the input signature"
            )
