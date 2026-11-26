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

"""GPU kernel launch configurations."""

from dataclasses import dataclass
from operator import index


@dataclass(frozen=True, slots=True)
class LaunchConfig:
    grid_dim: tuple[int, int, int] = (1, 1, 1)
    block_dim: tuple[int, int, int] = (1, 1, 1)
    shared_memory_bytes: int = 0

    def __post_init__(self) -> None:
        for name, dimensions in (
            ("grid_dim", self.grid_dim),
            ("block_dim", self.block_dim),
        ):
            try:
                normalized = tuple(index(dimension) for dimension in dimensions)
            except TypeError as error:
                raise TypeError(f"{name} dimensions must be integers") from error

            if len(normalized) != 3:
                raise ValueError(f"{name} must have exactly three dimensions")

            if any(dimension <= 0 for dimension in normalized):
                raise ValueError(f"{name} dimensions must be positive")

            object.__setattr__(self, name, normalized)

        try:
            shared_memory_bytes = index(self.shared_memory_bytes)
        except TypeError as error:
            raise TypeError("shared_memory_bytes must be an integer") from error

        if shared_memory_bytes < 0:
            raise ValueError("shared_memory_bytes must be nonnegative")

        object.__setattr__(self, "shared_memory_bytes", shared_memory_bytes)
