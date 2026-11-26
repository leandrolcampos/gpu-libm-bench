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

"""Tests for experiment library eligibility and selection."""

from pathlib import Path
from unittest.mock import Mock

import pytest
from hydra import compose, initialize_config_module

from gpu_libm_bench.compiler.libraries import (
    LibraryUnavailableError,
    ResolvedLibrary,
)
from gpu_libm_bench.compiler.targets import CompilationTarget
from gpu_libm_bench.compiler.toolchains import CUDAToolkit, LLVMToolchain
from gpu_libm_bench.configs import (
    ExperimentConfig,
    register_config_schema,
    validate_experiment_config,
)
from gpu_libm_bench.experiment import _libraries


@pytest.fixture
def config() -> ExperimentConfig:
    register_config_schema()
    with initialize_config_module(config_module="gpu_libm_bench.configs.yaml"):
        return validate_experiment_config(
            compose(
                config_name="experiment_config",
                overrides=["function=expf", "metric=latency", "backend=cuda"],
            )
        )


@pytest.fixture
def select(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    llvm = LLVMToolchain(
        tmp_path,
        tmp_path / "clang++",
        tmp_path / "llvm-dis",
        tmp_path / "llvm-nm",
        (tmp_path / "lib",),
        "test LLVM",
    )
    target = CompilationTarget("cuda", "nvptx64-nvidia-cuda", "sm_120")
    backend_sdk = CUDAToolkit(tmp_path / "cuda", tmp_path / "cuda/bin/ptxas", "13.4")

    def resolve(*, library_config, llvm, target, backend_sdk):
        return ResolvedLibrary(library_config.name)

    monkeypatch.setattr(_libraries, "resolve_library", resolve)

    def select(config):
        return _libraries.select_libraries(
            config, llvm=llvm, target=target, backend_sdk=backend_sdk
        )

    return select


def test_invalid_function_signature_is_rejected_before_preparing_libraries(
    config: ExperimentConfig,
    select,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config.function.signature.return_type = "double"
    resolve = Mock()
    monkeypatch.setattr(_libraries, "resolve_library", resolve)

    with pytest.raises(ValueError, match="return type must match"):
        select(config)

    resolve.assert_not_called()


@pytest.mark.parametrize("libraries", [None, ["libdevice", "llvm-libm"]])
def test_library_selection_preserves_order(
    config: ExperimentConfig, select, libraries: list[str] | None
) -> None:
    config.run.libraries = libraries

    assert [resolved_library.name for resolved_library in select(config)] == (
        libraries if libraries is not None else ["llvm-libm", "libdevice"]
    )


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize(
    ("missing", "cause"),
    [
        ("function", "does not declare function 'expf'"),
        ("bitcode", "bitcode is missing"),
    ],
)
def test_unavailable_library_is_skipped_only_for_automatic_selection(
    config: ExperimentConfig,
    select,
    monkeypatch: pytest.MonkeyPatch,
    explicit: bool,
    missing: str,
    cause: str,
) -> None:
    config.run.libraries = ["llvm-libm", "libdevice"] if explicit else None

    if missing == "function":
        config.libraries["llvm-libm"].functions.clear()
    else:
        monkeypatch.setattr(
            _libraries,
            "resolve_library",
            Mock(
                side_effect=[
                    LibraryUnavailableError(cause),
                    ResolvedLibrary("libdevice"),
                ]
            ),
        )

    if explicit:
        with pytest.raises(
            LibraryUnavailableError,
            match=f"library 'llvm-libm' is unavailable.*{cause}",
        ):
            select(config)
    else:
        assert [resolved_library.name for resolved_library in select(config)] == [
            "libdevice"
        ]


def test_bitcode_tool_failure_is_propagated(
    config: ExperimentConfig,
    select,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = RuntimeError("bitcode tool failed")
    monkeypatch.setattr(_libraries, "resolve_library", Mock(side_effect=error))

    with pytest.raises(RuntimeError) as caught:
        select(config)

    assert caught.value is error


def test_no_eligible_libraries_is_an_error(config: ExperimentConfig, select) -> None:
    for library_config in config.libraries.values():
        library_config.functions.clear()

    with pytest.raises(
        LibraryUnavailableError, match="no math libraries are available"
    ):
        select(config)
