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

"""Source and build artifact locations."""

from pathlib import Path

from gpu_libm_bench.backends import Backend


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


def backend_build_directory(
    backend: Backend | str, *, root: Path | None = None
) -> Path:
    return (backends_build_root() if root is None else root) / backend


def image_build_directory(
    library: str,
    *,
    backend: Backend | str,
    architecture: str,
    root: Path | None = None,
) -> Path:
    return (
        (images_build_root() if root is None else root)
        / backend
        / architecture
        / library
    )
