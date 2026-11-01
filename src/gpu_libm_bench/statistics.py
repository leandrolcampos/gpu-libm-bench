# Copyright 2026 Leandro A. Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Statistical estimators and confidence intervals for benchmark results."""

from operator import index

import numpy as np
from numpy.typing import NDArray


def mean_confidence_interval(
    values: NDArray[np.float64],
    *,
    resamples: int,
    confidence_level: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if isinstance(resamples, (bool, np.bool_)):
        raise TypeError("resamples must be an integer")
    try:
        resamples = index(resamples)
    except TypeError as error:
        raise TypeError("resamples must be an integer") from error

    if resamples < 1:
        raise ValueError("resamples must be positive")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")
    if values.ndim != 1 or values.size == 0:
        raise ValueError("values must be a nonempty one-dimensional array")
    if not np.isfinite(values).all():
        raise ValueError("values must be finite")

    means = np.empty(resamples, dtype=np.float64)
    for resample in range(resamples):
        indices = rng.integers(values.size, size=values.size)
        means[resample] = np.mean(values[indices])

    tail = (1 - confidence_level) / 2
    lower, upper = np.quantile(means, [tail, 1 - tail])
    return float(lower), float(upper)
