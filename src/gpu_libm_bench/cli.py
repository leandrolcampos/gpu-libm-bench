# Copyright 2026 Leandro A. Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Command-line interface for configuring and running benchmarks."""

import hydra
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException

from .configs import register_config_schema, validate_config

register_config_schema()


@hydra.main(config_path="configs/yaml", config_name="config")
def main(config: DictConfig) -> None:
    try:
        validate_config(config)
    except (ValueError, OmegaConfBaseException) as error:
        raise SystemExit(f"invalid experiment configuration: {error}") from error

    print("Configuration validated. Benchmark execution is not available yet.")
    print(OmegaConf.to_yaml(config, resolve=True))


if __name__ == "__main__":
    main()
