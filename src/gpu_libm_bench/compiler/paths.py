# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Source and build artifact locations."""

from pathlib import Path


def package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def source_root() -> Path:
    root = package_root().parents[1]
    if (root / "pyproject.toml").is_file() and all(
        (root / directory).is_dir() for directory in ("backends", "cmake", "kernels")
    ):
        return root

    raise FileNotFoundError(
        "native project sources were not found; use an editable project checkout"
    )


def build_root() -> Path:
    return source_root() / "build"


def backends_build_root() -> Path:
    return build_root() / "backends"


def images_build_root() -> Path:
    return build_root() / "kernels"


def backend_build_directory(backend: str, *, root: Path | None = None) -> Path:
    return (backends_build_root() if root is None else root) / backend


def image_build_directory(
    library: str,
    *,
    backend: str,
    architecture: str,
    root: Path | None = None,
) -> Path:
    return (
        (images_build_root() if root is None else root)
        / backend
        / architecture
        / library
    )
