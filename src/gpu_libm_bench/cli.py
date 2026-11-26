# ===----------------------------------------------------------------------=== #
# Copyright 2026 Leandro Lacerda Campos
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Command-line interface for configuring and running benchmarks."""

import argparse
import sys

import hydra
from omegaconf import DictConfig

from .configs import register_config_schema, validate_experiment_config
from .experiment import Experiment
from .reporting.terminal import print_result

register_config_schema()


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--details", action="store_true")
    options, hydra_arguments = parser.parse_known_args()

    @hydra.main(config_path="configs/yaml", config_name="experiment_config")
    def run(config: DictConfig) -> None:
        _run_experiment(config, show_details=options.details)

    original_arguments = sys.argv
    try:
        sys.argv = [original_arguments[0], *hydra_arguments]
        run()
    finally:
        sys.argv = original_arguments


def _run_experiment(config: DictConfig, *, show_details: bool) -> None:
    experiment_config = validate_experiment_config(config)
    result = Experiment(experiment_config).run()
    print_result(result, show_details=show_details)


if __name__ == "__main__":
    main()
