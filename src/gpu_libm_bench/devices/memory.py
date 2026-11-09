# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Typed device memory resources."""

from typing import TYPE_CHECKING, Generic, TypeVar, cast

import numpy as np
from numpy.typing import DTypeLike

from gpu_libm_bench.types import Scalar

if TYPE_CHECKING:
    from gpu_libm_bench.backends import _cuda  # pyright: ignore[reportMissingModuleSource]


T = TypeVar("T", bound=Scalar)


class DeviceBuffer(Generic[T]):  # noqa: UP046 -- Keep the scalar type invariant.
    def __init__(self, native: "_cuda.DeviceBuffer", dtype: np.dtype[T]) -> None:
        self._native = native
        self._dtype = dtype

    @property
    def dtype(self) -> np.dtype[T]:
        return self._dtype

    @property
    def size(self) -> int:
        return self.nbytes // self.dtype.itemsize

    @property
    def nbytes(self) -> int:
        return self._native.nbytes


def _buffer_dtype(dtype: DTypeLike) -> np.dtype[Scalar]:
    resolved = np.dtype(dtype)
    if resolved.kind not in "biufc" or not resolved.isnative:
        raise TypeError(
            f"device buffers require a native numeric dtype, got {resolved}"
        )
    return cast("np.dtype[Scalar]", resolved)
