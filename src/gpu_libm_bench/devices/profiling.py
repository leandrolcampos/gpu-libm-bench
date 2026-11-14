# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Kernel profiling metrics and results."""

from dataclasses import dataclass
from enum import Enum, auto


class KernelMetric(Enum):
    CLOCK_FREQUENCY = auto()
    DURATION = auto()


@dataclass(frozen=True, slots=True)
class KernelProfile:
    clock_frequency_hz: float | None = None
    duration_ms: float | None = None
