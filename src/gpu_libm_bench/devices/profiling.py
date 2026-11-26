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
