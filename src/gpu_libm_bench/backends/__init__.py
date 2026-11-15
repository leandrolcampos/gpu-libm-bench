# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Backend identities and native GPU execution backends."""

from enum import StrEnum


class Backend(StrEnum):
    CUDA = "cuda"
    HIP = "hip"
