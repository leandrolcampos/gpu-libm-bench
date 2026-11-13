# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from operator import index


@dataclass(frozen=True, slots=True)
class LaunchConfig:
    grid: tuple[int, int, int] = (1, 1, 1)
    block: tuple[int, int, int] = (1, 1, 1)
    shared_memory_bytes: int = 0

    def __post_init__(self) -> None:
        for name, dimensions in (("grid", self.grid), ("block", self.block)):
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
