# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Tests for native backend compilation selection."""

from pathlib import Path
from typing import cast

import pytest

from gpu_libm_bench.compiler.backends import configure_backend
from gpu_libm_bench.compiler.toolchains import CUDAToolkit


@pytest.mark.parametrize(
    ("backend", "sdk", "error", "message"),
    [
        ("cuda", object(), TypeError, "unsupported SDK type: object"),
        (
            "rocm",
            CUDAToolkit(Path("/cuda"), Path("/cuda/bin/ptxas"), "13.4"),
            ValueError,
            "non-CUDA backend",
        ),
    ],
)
def test_backend_configuration_rejects_incompatible_sdk(
    backend: str, sdk: object, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        configure_backend(backend, sdk=cast(CUDAToolkit, sdk))
