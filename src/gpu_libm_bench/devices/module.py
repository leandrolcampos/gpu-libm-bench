# Copyright 2026 Leandro A. Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""GPU device images and modules."""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gpu_libm_bench.compiler.targets import CompilationTarget


@dataclass(frozen=True, slots=True)
class DeviceImage:
    path: Path
    target: "CompilationTarget"
    library: str
    functions: tuple[str, ...] | None
    manifest: Path | None
