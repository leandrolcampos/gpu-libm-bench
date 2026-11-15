# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Backend identities and native GPU execution backends."""

from enum import StrEnum
from typing import Self


class Backend(StrEnum):
    CUDA = "cuda"
    HIP = "hip"

    @classmethod
    def _missing_(cls, value: object) -> Self:
        raise ValueError(f"unsupported backend: {value!r}")
