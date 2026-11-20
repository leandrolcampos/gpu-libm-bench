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


def test_experiment_compares_libraries_and_reuses_their_images(
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
                    "run.libraries=[llvm-libm,libdevice]",
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
    images = [result.image_build_info.image for result in first.results]
    assert [image.library for image in images] == ["llvm-libm", "libdevice"]

    timestamps = [image.image_path.stat().st_mtime_ns for image in images]
    second = Experiment(config).run()

    assert [image.image_path.stat().st_mtime_ns for image in images] == timestamps
    assert first.config.backend.architecture == first.device_info.architecture

    for before, after in zip(first.results, second.results, strict=True):
        assert before.image_build_info == after.image_build_info
        assert list(before) == ["central", "wide"]
        for result in before.values():
            assert result.measurement.elapsed_ticks.shape == (3, 3)
            assert np.isfinite(result.mean)
