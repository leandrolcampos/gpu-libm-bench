# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Command-line interface for configuring and running benchmarks."""

import argparse
import sys

import hydra
from omegaconf import DictConfig

from .configs import register_config_schema, validate_config
from .experiment import Experiment
from .reporting.terminal import print_result

register_config_schema()


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--details", action="store_true")
    options, hydra_arguments = parser.parse_known_args()

    @hydra.main(config_path="configs/yaml", config_name="config")
    def run(config: DictConfig) -> None:
        _run_experiment(config, show_details=options.details)

    original_arguments = sys.argv
    try:
        sys.argv = [original_arguments[0], *hydra_arguments]
        run()
    finally:
        sys.argv = original_arguments


def _run_experiment(config: DictConfig, *, show_details: bool) -> None:
    experiment_config = validate_config(config)
    result = Experiment(experiment_config).run()
    print_result(result, show_details=show_details)


if __name__ == "__main__":
    main()
