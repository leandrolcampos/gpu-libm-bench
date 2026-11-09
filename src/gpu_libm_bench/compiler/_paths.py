# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Source and build roots for native compilation."""

from pathlib import Path


def source_root() -> Path:
    root = Path(__file__).resolve().parents[3]
    if (root / "pyproject.toml").is_file() and all(
        (root / directory).is_dir() for directory in ("backends", "cmake", "kernels")
    ):
        return root

    raise FileNotFoundError(
        "native project sources were not found; use an editable project checkout"
    )


def build_root() -> Path:
    return source_root() / "build"
