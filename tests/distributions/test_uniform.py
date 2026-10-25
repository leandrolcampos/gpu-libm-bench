# Copyright 2026 Leandro A. Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Tests for the uniform distribution."""

from unittest.mock import Mock

import numpy as np
import pytest

from gpu_libm_math.distributions import Uniform
from gpu_libm_math.types import FloatType, SampleSize


@pytest.fixture(
    params=(np.float16, np.float32, np.float64), ids=("float16", "float32", "float64")
)
def dtype(request: pytest.FixtureRequest) -> FloatType:
    return request.param


@pytest.mark.parametrize(
    ("low", "high", "expected"),
    [
        (-10.0, 10.50, "uniform(-10, 10.5)"),
        (1.0000001, 1.0000002, "uniform(1.0000001, 1.0000002)"),
    ],
)
def test_string_representation(low: float, high: float, expected: str) -> None:
    assert str(Uniform(low=low, high=high, dtype=np.float64)) == expected


@pytest.mark.parametrize(
    ("sample_size", "shape"),
    [(None, None), (5, (5,)), ((2, 3), (2, 3)), ((), ()), (0, (0,))],
)
def test_sampling_preserves_size_and_dtype(
    dtype: FloatType, sample_size: SampleSize, shape: tuple[int, ...] | None
) -> None:
    distribution = Uniform(low=-10, high=10, dtype=dtype)

    samples = distribution(np.random.default_rng(42), sample_size=sample_size)

    if shape is None:
        assert isinstance(samples, dtype)
    else:
        assert isinstance(samples, np.ndarray)
        assert samples.shape == shape
        assert samples.dtype == np.dtype(dtype)


def test_repeated_sampling_is_reproducible(dtype: FloatType) -> None:
    distribution = Uniform(low=-10, high=10, dtype=dtype)
    first_rng = np.random.default_rng(123)
    second_rng = np.random.default_rng(123)

    first = distribution(first_rng, sample_size=128)
    second = distribution(first_rng, sample_size=128)

    np.testing.assert_array_equal(first, distribution(second_rng, sample_size=128))
    np.testing.assert_array_equal(second, distribution(second_rng, sample_size=128))
    assert not np.array_equal(first, second)


@pytest.mark.parametrize(
    ("low", "high", "expected"),
    [(2, 6, [2, 3, 4, 5]), (-2, 6, [-2, 0, 2, 4])],
)
def test_sampling_transforms_unit_interval(
    dtype: FloatType,
    low: float,
    high: float,
    expected: list[float],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "gpu_libm_math.distributions.uniform.random",
        Mock(return_value=np.array([0, 0.25, 0.5, 0.75], dtype=dtype)),
    )
    distribution = Uniform(low=low, high=high, dtype=dtype)

    samples = distribution(np.random.default_rng(42), sample_size=4)

    np.testing.assert_array_equal(samples, np.array(expected, dtype=dtype))


@pytest.mark.parametrize(
    ("dtype", "low", "high"), [(np.float16, 0.1, 0.3), (np.float32, 0.7, 1.1)]
)
def test_nonrepresentable_bounds_are_rounded_inward(
    dtype: FloatType, low: float, high: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "gpu_libm_math.distributions.uniform.random",
        Mock(return_value=np.array([0, np.nextafter(dtype(1), dtype(0))], dtype=dtype)),
    )
    distribution = Uniform(low=low, high=high, dtype=dtype)

    samples = distribution(np.random.default_rng(42), sample_size=2).astype(np.float64)

    assert np.all(samples >= low)
    assert np.all(samples <= high)


def test_full_finite_range_avoids_overflow(
    dtype: FloatType, monkeypatch: pytest.MonkeyPatch
) -> None:
    maximum = float(np.finfo(dtype).max)
    monkeypatch.setattr(
        "gpu_libm_math.distributions.uniform.random",
        Mock(return_value=np.array([0, np.nextafter(dtype(1), dtype(0))], dtype=dtype)),
    )

    with np.errstate(over="raise", invalid="raise"):
        distribution = Uniform(low=-maximum, high=maximum, dtype=dtype)
        samples = distribution(np.random.default_rng(42), sample_size=2).astype(
            np.float64
        )

    assert np.all(samples >= -maximum)
    assert np.all(samples <= maximum)


@pytest.mark.parametrize(("low", "high"), [(1, 1), (2, 1), (np.nan, 1), (0, np.inf)])
def test_invalid_bounds_are_rejected(low: float, high: float) -> None:
    with pytest.raises(ValueError):
        Uniform(low=low, high=high, dtype=np.float64)


def test_nonnumeric_bounds_are_rejected() -> None:
    with pytest.raises(TypeError):
        Uniform(low="0", high=1, dtype=np.float64)


def test_unsupported_dtype_is_rejected() -> None:
    with pytest.raises(TypeError):
        Uniform(low=0, high=1, dtype=np.int32)


def test_bounds_that_overflow_dtype_are_rejected() -> None:
    with pytest.raises(ValueError):
        Uniform(low=-1e10, high=1e10, dtype=np.float16)


def test_interval_without_representable_values_is_rejected() -> None:
    with pytest.raises(ValueError):
        Uniform(low=1.0001, high=1.0002, dtype=np.float16)
