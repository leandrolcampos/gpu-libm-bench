# Copyright 2026 Leandro A. Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Experiment schemas and consistency checks between configuration selections."""

from dataclasses import dataclass
from typing import Any, cast

from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

DistributionConfig = dict[str, Any]
DomainConfig = dict[str, DistributionConfig]


@dataclass
class ArgumentConfig:
    name: str
    type: str
    role: str


@dataclass
class SignatureConfig:
    return_type: str
    arguments: list[ArgumentConfig]


@dataclass
class FunctionConfig:
    name: str
    signature: SignatureConfig
    wrapper: str | None
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
class DeviceLibraryConfig:
    bitcode_path: str | None


@dataclass
class LibraryConfig:
    name: str
    backends: dict[str, DeviceLibraryConfig]


@dataclass
class BackendConfig:
    name: str
    target_triple: str
    gpu_arch: str | None
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
        provider="gpu-libm-math",
        replace=True,
    )
    ConfigStore.instance().store(
        group="schema/metric",
        name="latency",
        node=LatencyConfig,
        package="metric",
        provider="gpu-libm-math",
        replace=True,
    )


def validate_config(config: DictConfig) -> ExperimentConfig:
    structured = OmegaConf.merge(OmegaConf.structured(ExperimentConfig), config)
    experiment = cast("ExperimentConfig", OmegaConf.to_object(structured))

    domains = experiment.function.domains
    if not domains:
        raise ValueError("function.domains must contain at least one domain.")
    if any(not name.strip() for name in domains):
        raise ValueError("Domain names must not be empty.")

    selected = experiment.run.domains
    if selected is not None:
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("run.domains must be a nonempty list without duplicates.")
        for name in selected:
            if name not in domains:
                raise ValueError(f"run.domains contains an unknown domain: {name!r}.")

    if experiment.backend.name not in experiment.library.backends:
        raise ValueError(
            f"Library {experiment.library.name!r} does not declare support for "
            f"backend {experiment.backend.name!r}."
        )
    return experiment
