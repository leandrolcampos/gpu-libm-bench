# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for CUDA experiments and incremental image builds."""

from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_module

from gpu_libm_bench.configs import register_config_schema, validate_experiment_config
from gpu_libm_bench.experiment import Experiment
from gpu_libm_bench.experiment import experiment as experiment_module

pytestmark = [pytest.mark.cuda, pytest.mark.execution]


def test_experiment_collects_latency_and_reuses_its_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    register_config_schema()
    with initialize_config_module(config_module="gpu_libm_bench.configs.yaml"):
        config = validate_experiment_config(
            compose(
                config_name="experiment_config",
                overrides=[
                    "function=expf",
                    "metric=latency",
                    "backend=cuda",
                    "run.libraries=[llvm-libm]",
                    "metric.sample_size=3",
                    "metric.repetitions=3",
                    "metric.chain_length=32",
                    "metric.warmup_runs=1",
                ],
            )
        )

    monkeypatch.setattr(
        experiment_module.paths, "build_root", lambda: tmp_path / "build"
    )

    first = Experiment(config).run()
    image = first.results[0].image_build_info.image
    timestamp = image.image_path.stat().st_mtime_ns

    second = Experiment(config).run()

    assert image.image_path.stat().st_mtime_ns == timestamp
    assert first.results[0].image_build_info == second.results[0].image_build_info

    assert list(first.results[0]) == ["central", "wide"]
    assert first.config.backend.architecture == first.device_info.architecture

    for result in first.results[0].values():
        assert result.measurement.elapsed_ticks.shape == (3, 3)
        assert np.isfinite(result.mean)
