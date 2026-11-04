# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Tests for mean confidence intervals."""

import numpy as np
import pytest

from gpu_libm_bench.statistics import compute_mean_confidence_interval


@pytest.mark.parametrize(
    ("parameters", "error", "message"),
    [
        ({"resamples": 0}, ValueError, "resamples must be positive"),
        ({"resamples": 1.5}, TypeError, "resamples must be an integer"),
        ({"confidence_level": 1}, ValueError, "confidence_level must be between"),
        (
            {"confidence_level": float("nan")},
            ValueError,
            "confidence_level must be between",
        ),
        (
            {"values": np.array([], dtype=np.float64)},
            ValueError,
            "nonempty one-dimensional",
        ),
        ({"values": np.array([np.inf])}, ValueError, "values must be finite"),
    ],
)
def test_invalid_arguments_are_rejected(parameters, error, message) -> None:
    arguments = dict(
        values=np.array([1, 2], dtype=np.float64),
        resamples=10,
        confidence_level=0.95,
        rng=np.random.default_rng(0),
    )
    arguments.update(parameters)

    with pytest.raises(error, match=message):
        compute_mean_confidence_interval(**arguments)


def test_single_observation_returns_degenerate_interval() -> None:
    interval = compute_mean_confidence_interval(
        np.array([7.5], dtype=np.float64),
        resamples=10,
        confidence_level=0.95,
        rng=np.random.default_rng(0),
    )

    assert interval == (7.5, 7.5)
